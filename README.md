# cs492i_cv_team13
KAIST &amp; Naver [nsml] fashion dataset deep learning


This project is done by two ignorant KAIST students (1 graduate & 1 undergraduate)
for the class of CS492 - Deep Learning Project [Vision classification Task]
Thanks to cooperation from NAVER, NSML Platform (enables usage of GPUs and CPUs on the server)
 - https://ai.nsml.navercorp.com/intro


This README.md will help you with how to run our code using NSML platform

1. You will need to either download or pull all the codes in the repositry
2. Install NSML following the official document from NAVER : https://n-clair.github.io/ai-docs/_build/html/en_US/index.html
3. We two have used different programs to approach nsml server. Ubuntu and windows cmd with nsml installed according to the document above.
4. Upload our files to whichever program you prefer, and login to nsml account by typing $nsml login
5. type your ID and PW.
6. assuming you are allocated with at least one GPU, type $nsml run -d fashion_dataset -e main_MT_TSA_transform.py
since you are using fashion_dataset dataset already uploaded on nsml server publicly.
7. Done!

General Help with what each files contain:

setup.py -contains prerequisite libraries if used additionally to what nsml provides by default

models.py - contains basic training models - [resnet18, resnet50, densenet121]
            Setting for pretrained is turned False, because we want to see how much our code can improve training acc, 
            not how finely we can tune the pretrained models given.
            
ImageDataLoader_MT.py - Slight change to oringinally given baseline ImageDataLoader.py, which introduced parameter k to
                        SimpleImageloader function. The k will allow augmentation to the training data to increase the number of
                        training dataset. This file basically allows the main file to pass on train/valid/test loader to the
                        train() function.

main_MT_TSA_transform.py - Changed from originally given baseline main.py. We added MeanTeacher method to training function,
                           Introduced Time Signal Annealing technique in training function to load and train each epochs with 
                           different datasets according to randomized transform selection function newly created as well.
