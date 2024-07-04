import torch
from sklearn.cluster import KMeans
import numpy as np
from torchvision import transforms
import clip
import umap
from sklearn.decomposition import PCA
import open_clip
import hdbscan



device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
# model, preprocess = clip.load('ViT-B/32', device=device)
to_pil = transforms.ToPILImage()


def get_clip_features(images):
    image_list = torch.unbind(images)
    preprocessed_images = []
    for image in image_list:
        pil_image = to_pil(image)
        preprocessed_images.append(preprocess(pil_image).to(device))
    images = torch.stack(preprocessed_images)
    print(images.size(), images.get_device())
    with torch.no_grad():
        image_features = model.encode_image(images)

    return image_features.cpu().numpy()

def perform_kmeans(image_features, n):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(image_features)
    return kmeans.labels_

def separate_clusters(labels, n):
    cluster_indices = []
    for i in range(n):
        cluster_indices.append(np.where(labels==i)[0].tolist())

    return cluster_indices

def get_clip_clusters_kmeans(images, n=2, pre_indices=None):
    # images is expected to be a 4D tensor where each element is a 3D tensor representing an image
    # pre_indices allows for performing clustering on a larger cluster
    if pre_indices == None:
        pre_indices = list(range(len(images)))
    print('cluster pre_indices:', pre_indices)
    pre_indices = torch.tensor(pre_indices)
    images = torch.index_select(images, 0, pre_indices)

    clip_features = get_clip_features(images)
    # reduced_features = reduce_dimensions(clip_features)
    # labels = perform_kmeans(reduced_features, n)
    labels = perform_kmeans(clip_features, n)
    cluster_indices = separate_clusters(labels, n)

    print("Cluster 1 indices:", cluster_indices[0])
    print("Cluster 2 indices:", cluster_indices[1])

    return [[pre_indices[i] for i in cluster_index] for cluster_index in cluster_indices]
    # return [images[cluster_index].cuda() for cluster_index in cluster_indices]

def perform_hdbscan(image_features, min_cluster_size=2):
    # print(image_features.size(), min_cluster_size)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit(image_features)
    return clusterer.labels_

def get_clip_clusters_hdbscan(images, pre_indices=None, min_cluster_size=2):
    # images is expected to be a 4D tensor where each element is a 3D tensor representing an image
    # pre_indices allows for performing clustering on a larger cluster
    if pre_indices == None:
        pre_indices = list(range(len(images)))
    print('cluster pre_indices:', pre_indices)
    pre_indices = torch.tensor(pre_indices)
    images = torch.index_select(images, 0, pre_indices)

    clip_features = get_clip_features(images)
    labels = perform_hdbscan(clip_features, min_cluster_size)

    unique_labels = set(labels)
    print('Number of clusters:', len(unique_labels) - (1 if -1 in labels else 0))

    cluster_indices = []
    for label in unique_labels:
        cluster_indices.append(np.where(labels == label)[0].tolist())

    print([f"Cluster {i+1} indices:" + str(cluster_indices[i]) for i in range(len(cluster_indices))])

    return [[pre_indices[i] for i in cluster_index] for cluster_index in cluster_indices]

# def reduce_dimensions(image_features, n_components=3):
#     '''PCA'''
#     pca_model = PCA(n_components=n_components)
#     reduced_features = pca_model.fit_transform(image_features)
    
#     return reduced_features


def reduce_dimensions(image_features, n_components=5, n_neighbors=7, n_epochs=300, min_dist=0.1, random_state=35):
    '''UMAP'''
    dimension_model = umap.UMAP(n_neighbors=n_neighbors,
                                 n_epochs=n_epochs,
                                 min_dist=min_dist,
                                 n_components=n_components,
                                 random_state=random_state)
    reduced_features = dimension_model.fit_transform(image_features)
    return reduced_features



