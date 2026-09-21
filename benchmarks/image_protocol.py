"""Train-only, shared image-feature protocols. Labels never enter feature fitting."""
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


def prepare_images(dataset, output, download=False, reference_dim=64):
    from open_deep_tda.image_datasets import load_coil20, load_fashion_mnist
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    prepared=root/'prepared.json'
    if prepared.exists():
        metadata=json.loads(prepared.read_text())
        if metadata['dataset']!=dataset or metadata['reference_dim']!=reference_dim:
            raise ValueError('prepared data protocol differs; choose another directory')
        for name in ('features.npz','annotations.npz','reference_pca.npz','preprocessing_audit.npz'):
            if not (root/name).is_file():raise ValueError('incomplete prepared files; choose a fresh directory')
        return metadata
    started=time.perf_counter()
    if dataset=='coil20':
        data=load_coil20(download=download,image_size=32)
        views=np.rint(data['angles_degrees']/5).astype(int)
        test_mask=views%3==0
        train_images=data['images'][~test_mask];test_images=data['images'][test_mask]
        labels_train=data['object_ids'][~test_mask];labels_test=data['object_ids'][test_mask]
        annotation={'train_labels':labels_train,'test_labels':labels_test,
                    'object_ids':np.concatenate([labels_train,labels_test]),
                    'angles_degrees':np.concatenate([data['angles_degrees'][~test_mask],data['angles_degrees'][test_mask]]),
                    'original_ids':np.concatenate([np.flatnonzero(~test_mask),np.flatnonzero(test_mask)])}
        split='20 objects, 48 training + 24 held-out views each; every third 5-degree view held out; angle/object never given to reducers'
    elif dataset=='fashion_mnist':
        data=load_fashion_mnist(download=download)
        train_images,test_images=data['train_images'],data['test_images']
        annotation={'train_labels':data['train_labels'],'test_labels':data['test_labels']}
        split='full official 60000/10000 split; no subsampling of fit or test transforms'
    else:raise ValueError('dataset must be coil20 or fashion_mnist')
    expected=(960,480) if dataset=='coil20' else (60000,10000)
    if (len(train_images),len(test_images))!=expected:raise ValueError('unexpected dataset size; not silently accepting a reduced dataset')
    Xtrain=train_images.reshape(len(train_images),-1).astype(np.float32)/255.
    Xtest=test_images.reshape(len(test_images),-1).astype(np.float32)/255.
    rng=np.random.default_rng(2026)
    audit_ids=np.sort(rng.choice(len(Xtest),min(512,len(Xtest)),replace=False))
    audit_pixels=Xtest[audit_ids].copy()
    pca=PCA(n_components=reference_dim,svd_solver='randomized',iterated_power=4,random_state=2026)
    with threadpool_limits(limits=1):
        pca.fit(Xtrain)
        # Use the same persisted linear map for train and test, rather than
        # randomized fit_transform's approximate factorization scores.
        Htrain=pca.transform(Xtrain).astype(np.float32)
        Htest=pca.transform(Xtest).astype(np.float32)
    pairs=rng.integers(len(Htrain),size=(20000,2))
    distances=np.linalg.norm(Htrain[pairs[:,0]].astype(float)-Htrain[pairs[:,1]],axis=1)
    scale=float(np.median(distances[distances>0]))
    train=np.asarray(Htrain/scale,dtype=np.float32);test=np.asarray(Htest/scale,dtype=np.float32)
    if not np.isfinite(train).all() or not np.isfinite(test).all():raise ValueError('nonfinite image reference features')
    digest=hashlib.sha256(train.tobytes()+test.tobytes()).hexdigest()
    np.savez_compressed(root/'features.npz',train=train,test=test)
    np.savez_compressed(root/'annotations.npz',**annotation)
    np.savez_compressed(root/'reference_pca.npz',components=pca.components_,mean=pca.mean_,distance_scale=np.array(scale))
    np.savez_compressed(root/'preprocessing_audit.npz',raw_pixels=audit_pixels,pca_features=Htest[audit_ids],test_ids=audit_ids)
    metadata={'dataset':dataset,'source':data['metadata'],'split':split,'n_train':len(train),'n_test':len(test),
        'pixel_feature_dim':Xtrain.shape[1],'reference_dim':reference_dim,'reference_type':'fixed train-only PCA, NOT a trained image SSL encoder',
        'pca_solver':'randomized, iterated_power=4, random_state=2026, whiten=False',
        'pca_training_explained_variance':float(pca.explained_variance_ratio_.sum()),
        'train_only_distance_scale':scale,'feature_sha256':digest,
        'preparation_seconds':time.perf_counter()-started,
        'preprocessing_audit_scope':'fixed held-out image subset; raw pixels versus unscaled PCA features; not a full raw-image topology guarantee',
        'labels_usage':'evaluation/split annotations only, not PCA or reducer training'}
    temporary=prepared.with_suffix('.tmp');temporary.write_text(json.dumps(metadata,indent=2,allow_nan=False));temporary.replace(prepared)
    del Xtrain,Xtest,Htrain,Htest,train,test,data,train_images,test_images
    gc.collect()
    return metadata
