# Download the Intentonomy Dataset

![ontology](images/ontology.gif)

Intentonomy dataset contains Intentonomy dataset has 12,740 training, 498 val, 1217 test images. Each image contains one or multiple intent categories. We select 28 labels from a general human motive taxonomy used in [psychology research](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0172279). There are 9 super-categories in total (in black box), namely “virtues”, “self-fulfill”, “openness to experience”, “security and belonging”, “power”, “health”, “family”, “ambition and ability”, ”financial and occupational success”.

## Download the data

> :exclamation:**NOTE**: The val/test split of this released version is **different** than the *val / split* used in the paper.  We notice that the *test* split used in the paper does not contain one intent label. Therefore, we (1) re-arranged the images to ensure each intent labels occurs both *val* and *test* splits, and (2) further cleaned the label for both split. The updated *val/test* splits do not affect the conclusion made by the paper. See [here](https://github.com/KMnP/intentonomy#baseline-results) for the **updated results**. 



### Step 1: Get the annotation files

|       | Image Number | Link                                                         |
| ----- | ------------ | ------------------------------------------------------------ |
| Train | 12,740       | [intentonomy_train2020.json](https://cornell.box.com/s/rff4fuq20t7tc4edx2wl0golh64zf9qh) |
| Val   | 498          | [intentonomy_val2020.json](https://cornell.box.com/s/3dmyavfpyyayxylfo9fzmaj1v2j0gqmk) |
| Test  | 1,217        | [intentonomy_test2020.json](https://cornell.box.com/s/3ep2w96qf91w9qvqop2ri95g0e3fx4zj) |

### Step 2: Get the images

We utilize a subset of `Unsplash Full Dataset 1.1.0`. We recommend to get the images according to the url we provided. Images will take approximately 20G disk space. 

Execute below script will downloads images from unsplash using multiple threads (half of cpu count of your machine).
Images that already exist will not be downloaded again, so the script can resume a partially completed download.

```bash
# ANN_ROOT=""
# IMG_ROOT=""
python download_images.py --anno-root $ANN_ROOT --image-root $IMG_ROOT
```

We also encourage you to check out [`Unsplash Full Dataset 1.1.0`](https://github.com/unsplash/datasets)  if you would like to use the provided metadata such as colors and keywords.

## Data Format

We follow the annotation format of the [COCO dataset](http://mscoco.org/dataset/#download) and the majority of [FGVC](http://fgvc.org/) challenges. The annotations are stored in the [JSON format](http://www.json.org/) and are organized as follows.

> :exclamation:**NOTE**: the soft probability in training annotations are the result of normalization across annotators. Out of the three annotators, if `x`of them think this image belong to one label with id `m`, then this label become `x/3` in the annotation at position `m`. Each intent label was annotated independently for all images. Thus the vector does not sum to 1. 

```bash
{
 "info": info,
 "categories": [category],
 "images": [image],
 "annotations": [annotation],
}

info{
  "year" : int,
  "version" : str,
  "description" : str,
  "date_created" : datetime,
}

category{
  "id" : int,
  "name" : str,
}

image{
  "id" : str,
  "filename" : str,
  "original_id": str,
  'unsplash_url': str,
}

annotation{
  "id" : int,
  "image_id" : str,
  "category_ids" : [int],            # in val/test annotations
  "category_ids_softprob": [float],  # in training annotations. 
}
```



## Terms of Use

### Annotations

The annotations in the Intentonomy dataset along with this repo, belong to the Intentonomy and are licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode).

### Images

Intentonomy does not own the copyright of the images. Use of the images must abide by the Terms of use of [Unsplash](https://unsplash.com/license). The users of the Intentonomy accept full responsibility for the use of the Intentonomy dataset, including but not limited to the use of any copies of copyrighted images that they may create from the Intentonomy.



## Citation

If you find our work helpful in your research, please cite it as:

```tex
@inproceedings{jia2021intentonomy,
  title={Intentonomy: a Dataset and Study towards Human Intent Understanding},
  author={Jia, Menglin and Wu, Zuxuan and Reiter, Austin and Cardie, Claire and Belongie, Serge and Lim, Ser-Nam},
  booktitle={CVPR},
  year={2021}
}
```
