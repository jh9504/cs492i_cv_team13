# cs492i_cv_team13
KAIST &amp; Naver [nsml] fashion dataset deep learning

## Usage
This project is done by two ignorant KAIST students (1 graduate & 1 undergraduate)
for the class of CS492 - Deep Learning Project [Vision classification Task]
Thanks to cooperation from NAVER, NSML Platform (enables usage of GPUs and CPUs on the server)
 - https://ai.nsml.navercorp.com/intro

## Description
This README.md will help you with how to run our code using NSML platform

1. You will need to either download or pull all the codes in the repositry
2. Install NSML following the official document from NAVER : https://n-clair.github.io/ai-docs/_build/html/en_US/index.html
3. We two have used different programs to approach nsml server. Ubuntu and windows cmd with nsml installed according to the document above.
4. Upload our files to whichever program you prefer, and login to nsml account by typing $nsml login
5. type your ID and PW.
6. assuming you are allocated with at least one GPU, type $nsml run -d fashion_dataset -e main_MT_TSA_transform.py
since you are using fashion_dataset dataset already uploaded on nsml server publicly.
7. Done!

## Files
General Help with what each files contain:

setup.py 
 -contains prerequisite libraries if used additionally to what nsml provides by default

models.py 
 - contains basic training models - [resnet18, resnet50, densenet121]
Setting for pretrained is turned False, because we want to see how much our code can improve training acc,
not how finely we can tune the pretrained models given.
            
ImageDataLoader_MT.py 
 - Slight change to oringinally given baseline ImageDataLoader.py, which introduced parameter k to
   SimpleImageloader function. The k will allow augmentation to the training data to increase the number of
   training dataset. This file basically allows the main file to pass on train/valid/test loader to the
   train() function.

main_MT_TSA_transform.py 
 - Changed from originally given baseline main.py. We added MeanTeacher method to training function,
 Introduced Time Signal Annealing technique in training function to load and train each epochs with 
 different datasets according to randomized transform selection using the function newly created as 
 well.
 The types of tests we offer are
  1. Different applications of basic settings(we recommend setting A,B,C defined in the code: best result settings we tested):   
   ```
   a) no_trainaug k - number of augmentation for each training data
   b) batchsize - training data batchsize
   c) unlbatchsize - unlabeled data batchsize
   d) epochdrop - epoch of which batch size drops
   e) tbs_d - batchsize for epoch drop
   f) utb_d - batchsize_unlabeled for eopch drop
   g) epochdropdrop - epoch of which batchsize drops again (second drop)
   h) tbs_dd - batchsize for epoch drops second time
   i) utb_dd - unlabeled data batchsize for epoch drops second time 
   ```                           
  2. Different Augmentation/transform types and number of transforms to be done
   ```
   a) randomize - if True, randomly select transform types and how many transforms to select
   b) n - if not 0, n number of transform types will be randomly selected
   c) resize_crop - resizes the image and crops a portion
   d) gray - changes the image in grayscale()
   e) horizontal - horizontally flips the image
   f) jitter - has 4 degree of changes to images(brightness, contrast, saturation, hue)
   g) rotate - rotates the image (-35~35 degrees)
   h) vertical - vertically flips the image (not recommended for this dataset, since shopping images are not intentionally inverted upside down
   ```
   
Good luck with the Tests!
