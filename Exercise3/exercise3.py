import os
import shlex
import argparse
from tqdm import tqdm

# for python3: read in python2 pickled files
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from parmap import parmap

def parseArgs(parser):
    parser.add_argument('--labels_test', 
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', 
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    parser.add_argument('--extract-sift', action='store_true',
                        help='extract SIFT descriptors and save to icdar2017-sift-test and icdar2017-sift-train folders')
    return parser

def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        
        # If file doesn't exist with the specified pattern, try common image extensions
        # This handles cases where train uses .png and test uses .jpg
        if not os.path.exists(true_file_name) and pattern in ['.png', '.jpg', '.jpeg']:
            # Try alternative image extensions
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                alt_file = os.path.join(folder, file_name + ext)
                if os.path.exists(alt_file):
                    true_file_name = alt_file
                    break
        
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def getImageFiles(folder, labelfile):
    """
    Get image files from labelfile, trying common image extensions
    parameters:
        folder: input folder
        labelfile: contains a list of filename and labels
    return: absolute filenames + labels
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    all_files = []
    labels = []
    for line in all_lines:
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # Try to find image file with common extensions
        found = False
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            img_file = os.path.join(folder, file_name + ext)
            if os.path.exists(img_file):
                all_files.append(img_file)
                labels.append(class_id)
                found = True
                break
        
        if not found:
            # If no image found, skip this entry
            continue

    return all_files, labels

def computeDescs(filename):
    """
    compute SIFT descriptors from an image file
    parameters:
        filename: path to image file
    returns: TxD matrix of descriptors (T descriptors of dimension D)
    """
    # Load image
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {filename}")
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints
    keypoints = sift.detect(img, None)
    
    # If no keypoints found, return empty array
    if len(keypoints) == 0:
        return np.array([]).reshape(0, 128)
    
    # Set all keypoint angles to 0
    for kp in keypoints:
        kp.angle = 0.0
    
    # Compute descriptors with modified keypoints (angle=0)
    keypoints, descriptors = sift.compute(img, keypoints)
    
    # If no descriptors found, return empty array
    if descriptors is None or len(descriptors) == 0:
        return np.array([]).reshape(0, 128)
    
    # Apply Hellinger normalization:
    # 1. L1 normalization (since SIFT descriptors are already L2 normalized)
    descriptors = descriptors.astype(np.float32)
    l1_norms = np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True)
    l1_norms[l1_norms == 0] = 1  # avoid division by zero
    descriptors = descriptors / l1_norms
    
    # 2. Sign square root (no L2 normalization afterwards)
    descriptors = np.sign(descriptors) * np.sqrt(np.abs(descriptors))
    
    return descriptors

def extractAndSaveSIFT(image_files, output_folder, suffix='_SIFT_patch_pr.pkl.gz'):
    """
    Extract SIFT descriptors from images and save them to output folder
    parameters:
        image_files: list of image file paths
        output_folder: folder to save descriptor files
        suffix: suffix to append to base filename
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for img_file in tqdm(image_files, desc='Extracting SIFT'):
        # Check if it's an image file
        is_image = any(img_file.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'])
        
        if not is_image:
            continue
        
        # Get base filename without extension
        base_name = os.path.basename(img_file)
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            if base_name.endswith(ext):
                base_name = base_name[:-len(ext)]
                break
        
        # Construct output filename
        output_file = os.path.join(output_folder, base_name + suffix)
        
        # Skip if already exists
        if os.path.exists(output_file):
            continue
        
        # Compute descriptors
        desc = computeDescs(img_file)
        
        # Save descriptors
        with gzip.open(output_file, 'wb') as fOut:
            cPickle.dump(desc, fOut, -1)

def loadRandomDescriptors(files, max_descriptors):
    """ 
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D (or image files)
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(min(max_files, len(files)))
    files = np.array(files)[indices]
   
    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        # Check if file is an image file or pickle file
        is_image = any(files[i].endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'])
        
        if is_image:
            # Compute descriptors from image
            desc = computeDescs(files[i])
        else:
            # Load from pickle file
            with gzip.open(files[i], 'rb') as ff:
                # for python2
                # desc = cPickle.load(ff)
                # for python3
                desc = cPickle.load(ff, encoding='latin1')
        
        # Skip if no descriptors found
        if len(desc) == 0:
            continue
            
        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[ indices ]
        descriptors.append(desc)
    
    if len(descriptors) == 0:
        raise ValueError("No descriptors found in any files")
    
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

def dictionary(descriptors, n_clusters):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    # TODO
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=10000,
        verbose=True
    )
    kmeans.fit(descriptors)
    return kmeans.cluster_centers_

def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    # compute nearest neighbors
    # TODO
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(descriptors, clusters, k=1) 

    # create hard assignment
    assignment = np.zeros( (len(descriptors), len(clusters)) )
    # TODO
    for i, m in enumerate(matches):
        k = m[0].trainIdx
        assignment[i, k] = 1.0

    return assignment

def vlad(files, mus, powernorm, gmp=False, gamma=1000):
    """
    compute VLAD encoding for each files
    parameters: 
        files: list of N files containing each T local descriptors of dimension
        D (or image files)
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]
    encodings = []

    for f in tqdm(files):
        # Check if file is an image file or pickle file
        is_image = any(f.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'])
        
        if is_image:
            # Compute descriptors from image
            desc = computeDescs(f)
        else:
            # Load from pickle file
            with gzip.open(f, 'rb') as ff:
                desc = cPickle.load(ff, encoding='latin1')
        
        # Skip if no descriptors found
        if len(desc) == 0:
            # Create zero encoding
            D = mus.shape[1]
            f_enc = np.zeros((D*K), dtype=np.float32)
            if powernorm:
                f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))
            f_enc = normalize(f_enc.reshape(1, -1), norm='l2').flatten()
            encodings.append(f_enc)
            continue
            
        a = assignments(desc, mus)
        
        T,D = desc.shape
        f_enc = np.zeros( (D*K), dtype=np.float32)
        for k in range(mus.shape[0]):
            # it's faster to select only those descriptors that have
            # this cluster as nearest neighbor and then compute the 
            # difference to the cluster center than computing the differences
            # first and then select
            mask = a[:, k] == 1
            if not np.any(mask):
                continue
            diff = desc[mask] - mus[k]
            encoding = diff.sum(axis=0)

            start = k * D
            end = (k + 1) * D
            f_enc[start:end] = encoding

   
        # c) power normalization
        if powernorm:
            # TODO
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

        # l2 normalization
        # TODO
        f_enc = normalize(f_enc.reshape(1, -1), norm='l2').flatten()

        encodings.append(f_enc)

    encodings = np.vstack(encodings)


    return encodings

def esvm(encs_test, encs_train, C=1000):
    """ 
    compute a new embedding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives   
    parameters: 
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """


    # set up labels
    # TODO
    M = encs_train.shape[0]
    def loop(i):
        # compute SVM 
        # and make feature transformation
        # TODO
        x_pos = encs_test[i:i+1]          
        X = np.vstack([x_pos, encs_train])  
        y = np.zeros(1 + M, dtype=int)
        y[0] = 1  

        clf = LinearSVC(C=C, class_weight='balanced')
        clf.fit(X, y)

        w = clf.coef_.astype(np.float32) 

        w_norm = np.linalg.norm(w)
        if w_norm > 0:
            w = w / w_norm

        return w

    # let's do that in parallel: 
    # if that doesn't work for you, just exchange 'parmap' with 'map'
    # Even better: use DASK arrays instead, then everything should be
    # parallelized
    new_encs = list(parmap( loop, tqdm(range(len(encs_test)))))
    new_encs = np.concatenate(new_encs, axis=0)
    # return new encodings
    return new_encs


def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    # TODO
    sims = np.dot(encs, encs.T).astype(np.float32)
    dists = 1.0 - sims
    # mask out distance with itself
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42) # fix random seed
    
    # If --extract-sift is set, extract descriptors and save them
    if args.extract_sift:
        print('> Extracting SIFT descriptors...')
        # Get image files for train and test (original images, not descriptors)
        train_images, _ = getImageFiles(args.in_train, args.labels_train)
        test_images, _ = getImageFiles(args.in_test, args.labels_test)
        
        # Use descriptor suffix for saving (default to _SIFT_patch_pr.pkl.gz)
        # If user provided a descriptor suffix, use it; otherwise use default
        desc_suffix = args.suffix
        if desc_suffix in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            # If suffix is an image extension, use default descriptor suffix
            desc_suffix = '_SIFT_patch_pr.pkl.gz'
        
        # Extract and save SIFT descriptors
        print('> Extracting SIFT for training images...')
        extractAndSaveSIFT(train_images, 'icdar2017-sift-train', suffix=desc_suffix)
        print('> Extracting SIFT for test images...')
        extractAndSaveSIFT(test_images, 'icdar2017-sift-test', suffix=desc_suffix)
        
        # Update input folders and suffix to use the extracted descriptors
        args.in_train = 'icdar2017-sift-train'
        args.in_test = 'icdar2017-sift-test'
        args.suffix = desc_suffix
   
    # a) dictionary
    files_train, labels_train = getFiles(args.in_train, args.suffix,
                                         args.labels_train)
    print('#train: {}'.format(len(files_train)))
    if not os.path.exists('mus.pkl.gz'):
        # TODO
        descriptors = loadRandomDescriptors(files_train, max_descriptors=500000)
        print('> loaded {} descriptors:'.format(len(descriptors)))

        # cluster centers
        print('> compute dictionary')
        # TODO
        mus = dictionary(descriptors, n_clusters=100)
        with gzip.open('mus.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)

  
    # b) VLAD encoding
    print('> compute VLAD for test')
    files_test, labels_test = getFiles(args.in_test, args.suffix,
                                       args.labels_test)
    print('#test: {}'.format(len(files_test)))

    gamma = args.gamma
    fname = 'enc_test_gmp{}.pkl.gz'.format(gamma) if args.gmp else 'enc_test.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        # TODO
        enc_test = vlad(files_test, mus, powernorm=args.powernorm,
                        gmp=args.gmp, gamma=gamma)  
        # ----------  evaluate without powernorm   ---------- 
        print('> evaluate VLAD encodings without power normalization')  
        enc_test_no_pnorm = vlad(files_test, mus, powernorm=False,
                        gmp=args.gmp, gamma=gamma)  
        evaluate(enc_test_no_pnorm, labels_test) 
        # ------------------------------------------------------ 
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)
   
    # cross-evaluate test encodings
    print('> evaluate VLAD encodings with power normalization')
    evaluate(enc_test, labels_test)

    


    # d) compute exemplar svms
    print('> compute VLAD for train (for E-SVM)')
    fname = 'enc_train_gmp{}.pkl.gz'.format(gamma) if args.gmp else 'enc_train.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        # TODO
        enc_train = vlad(files_train, mus, powernorm=args.powernorm,
                         gmp=args.gmp, gamma=gamma)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_train = cPickle.load(f)

    print('> esvm computation')
    # TODO
    enc_test = esvm(enc_test, enc_train, C=args.C)

    # eval
    evaluate(enc_test, labels_test)
    print('> evaluate')
