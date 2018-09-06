READ ME
------------------------------------------------------
Adverserial examples are generated when a small perturburation is added to a clean image. 
The adverserial example are then missclassified to the wrong output. The adverserial examples 
are used to fool a machine learning model.

In this project, we have generated adverserial examples for malware binaries i.e the binaries of
files which were classified as malware by a machine learning model have been perturbed by a small
noise, so that they evade a machine learning model and are missclassified to a wrong class of malware.

The following algorithms have been used to generate the adverserial example

1) JSMA 
2) FGSM
3) C&W attack
4) ATN
