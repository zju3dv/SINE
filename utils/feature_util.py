from sklearn.decomposition import PCA
import numpy as np


def pca_transform(image_features, pca=None, dim=3):
    assert np.any(np.isnan(image_features)) == False
    if pca is None:
        pca = PCA(n_components=dim)
        pca.fit(image_features)
    transformed_feat = pca.transform(image_features)

    return transformed_feat, pca


def normalize_feat_color(feature, uint8_color=True):
    min_feat = np.min(feature, axis=-1)[:, None]
    max_feat = np.max(feature, axis=-1)[:, None]
    transformed_feat = (feature - min_feat) / (max_feat - min_feat)

    if uint8_color == True:
        colors = (transformed_feat * 255).astype(np.uint8)
    else:
        colors = transformed_feat

    return colors
