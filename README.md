# representation-learning-methods
Repository containing representation learning methods. 

# Running experiments
## Download and prepare datasets:
```
cd data
bash download_and_prep_data.sh
```
## Latent Flow
Pretrain model:

```
python scripts/train_flow.py 
```

Train classifier:

```
python scripts/train_clf_flow.py --encoder_path pretrained/encpder/path.ckpt --classifier_path classifier/output/directory
```

Getting test results:
```
python scripts/test_clf_flow.py --module_path path/to/trained/model
```

## BiGan
Pretrain raw model:

```
python scripts/train_bigan.py 
```

Pretrain model pretrained on ImageNet:

```
python scripts/train_bigan_pretrained.py 
```

Getting results:
```
python scripts/test_bigan.py --path path/to/checkpoint
```

## Beta Vae
Train BVAE:
```
python scripts/b_vae_train.py
```

```
python scripts/test_b_vae.py
```
