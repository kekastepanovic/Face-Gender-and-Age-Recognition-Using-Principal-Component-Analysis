import numpy as np
import cv2
import os
import random2 as random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
img_list=list() # for images as 1-D vector
img_mf_list=list() #for male or female
img_age_list=list() # for older or younger than 45
test_mf_predicted=list() #for predictions
test_age_predicted=list() #for predictions
face_file_name_lista=list() # for name of pictures/files
correct_pred=0 # correct predicted male or female
correct_age=0 # correct predicted age
slicne_poredjenje=0 # for similar photos - test and train
folder=r"C:\Users\Computer\Desktop\KV\Viola and Jones"
folder_slicne_slike=r"C:\Users\Computer\Desktop\KV\treniranje i testiranje"
folder_jedinicna_lica=r"C:\Users\Computer\Desktop\KV\jedinicna lica"
num_components=50# number of Eigen faces
ii=0 # memory problem for too much pictures
while ii<=110:
   # we choose the photo for viola Jones application randomly
   file=random.choice([x for x in os.listdir(r"C:\Users\Computer\Desktop\KV\part1") if os.path.isfile(os.path.join(r"C:\Users\Computer\Desktop\KV\part1", x))])
   ii=ii+1
   original_image=cv2.imread(os.path.join(r"C:\Users\Computer\Desktop\KV\part1",file))
   grayscale_image=cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
   face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
   detected_faces=face_cascade.detectMultiScale(grayscale_image)
   i=0 # taking only the first detected face on the picture
   for(column,row,width,height) in detected_faces:
       cv2.rectangle(grayscale_image,(column,row),(column+width,row+height),(0,255,0),2)
       sub_face=grayscale_image[row:row+height,column:column+width]
       face_file_name=file
       sub_face=cv2.resize(sub_face,(360,480))
       if i==0:
          face_file_name_lista.append(face_file_name) # name of the photo
          img_list.append(sub_face) # list of crop pictures after Viola and Jones
          cv2.imwrite(os.path.join(folder,face_file_name),sub_face)
          male_or_female = face_file_name.find("_")
          age=face_file_name[0:male_or_female]
          age=int(age)
          if(age<45):
              img_age_list.append(0) # first group for age
          else:
              img_age_list.append(1) # second grou
          img_mf_list.append(face_file_name[male_or_female + 1]) # male or female
          cv2.waitKey(40)
       i=i+1
# we need Eigen faces
s=0
duzina=0 # number of pictures in a img_list
for img in img_list:
    duzina=duzina+1 # number of images in the list
image_matrix=np.zeros(shape=(duzina,172800)) # every row is one photo
for img in img_list: # every row is one photo
    image_matrix[s,:]=np.reshape(img,(1,172800),'F')
    s=s+1
train_index=np.random.rand(len(image_matrix))<0.8
img_mf_list=np.array(img_mf_list)
img_age_list=np.array(img_age_list)
face_file_name_lista=np.array(face_file_name_lista)
train_matrix=image_matrix[train_index,:]
face_file_name_lista_train=face_file_name_lista[train_index]
train_mf_indeks=img_mf_list[train_index]
train_age_indeks=img_age_list[train_index]
test_matrix=image_matrix[~train_index,:]
face_file_name_lista_test=face_file_name_lista[~train_index]
test_mf_indeks=img_mf_list[~train_index]
test_age_indeks=img_age_list[~train_index]
mean_img_train=np.sum(train_matrix,axis=0)/len(train_matrix)
mean_img_test=np.sum(test_matrix,axis=0)/len(test_matrix)
train_matrix_prev=train_matrix # before all the operation
test_matrix_prev=test_matrix
for xx in range(len(train_matrix)):
    train_matrix[xx,:]=train_matrix[xx,:]-mean_img_train
for xx in range(len(test_matrix)):
   test_matrix[xx,:]=test_matrix[xx,:]-mean_img_test
# Signular Value Decomposition matrix=U*Sigma*V
# colums of U are left -singular vectors, of V right singular vectors, diagonal elements of Sigma are singular values of matrix
# Sigma is already arranged in descending order as well as VT.T
U, Sigma, VT = np.linalg.svd(train_matrix, full_matrices=False)
# original-mean value is used
new_train_matrix=np.matmul(train_matrix,VT[:num_components,:].T) # projection in Eigen vectors space
# original-mean value is used
new_test_matrix=np.matmul(test_matrix,VT[:num_components,:].T) # projection in Eigen vectors space
# every row of matrix is one photo projected in Eigen face space
number_testing=len(new_test_matrix)
for r_idx in range(len(new_test_matrix)):
    if(slicne_poredjenje<=3):
          slika_test=np.reshape(test_matrix_prev[r_idx,:],(360,480),'C')
          ime_slike=str(slicne_poredjenje)+"test"+face_file_name_lista_test[r_idx]
          cv2.imwrite(os.path.join(folder_slicne_slike,ime_slike),slika_test)
    distances_euclidian = list()
    for training in range(len(new_train_matrix)):
       distances_euclidian.append(np.sum((new_train_matrix[training,:]-new_test_matrix[r_idx,:])**2,axis=0))
    image_closest = distances_euclidian.index(min(distances_euclidian))
    if (slicne_poredjenje <= 3):
        slika_test_train = np.reshape(train_matrix_prev[image_closest, :], (360, 480),'C')
        ime_slike = str(slicne_poredjenje)+"train"+face_file_name_lista_train[image_closest]
        cv2.imwrite(os.path.join(folder_slicne_slike, ime_slike), slika_test_train)
    morf=train_mf_indeks[image_closest]
    test_mf_predicted.append(morf)
    age=train_age_indeks[image_closest]
    test_age_predicted.append(age)
    if(test_mf_indeks[r_idx]==morf):
        correct_pred=correct_pred+1
    if(test_age_indeks[r_idx]==age):
        correct_age=correct_age+1
    del distances_euclidian[:]
    slicne_poredjenje=slicne_poredjenje+1 # da bi imali samo 4 slike koje smo uporedili
test_mf_predicted=np.array(test_mf_predicted)
test_age_predicted=np.array(test_age_predicted)
PCA_accuracy=(correct_pred/number_testing)*100
PCA_age=(correct_age/number_testing)*100
print(PCA_accuracy) # accuracy using PCA
print(confusion_matrix(test_mf_indeks,test_mf_predicted))
print(PCA_age)
print(confusion_matrix(test_age_indeks,test_age_predicted))
# Neural Network - MLP classifier
# hidden_layer_size=n_layers-2 - ith element represents the number of neurons in the ith layer
# activation-activation function
# solver - the solver for weight optimization
# batch_size - size of mini batches for stochastic optimizer
# early stopping - to prevent overfitting
clf=MLPClassifier(hidden_layer_sizes=(15,10),max_iter=1000,solver='adam',batch_size='auto',early_stopping=True)
clf.fit(new_train_matrix,train_mf_indeks)
y_pred=clf.predict(new_test_matrix)
print(confusion_matrix(test_mf_indeks,y_pred))
clf_age=MLPClassifier(hidden_layer_sizes=(15,10),max_iter=1000,solver='adam',batch_size='auto',early_stopping=True)
clf_age.fit(new_train_matrix,train_age_indeks)
y_pred_age=clf_age.predict(new_test_matrix)
print(confusion_matrix(test_age_indeks,y_pred_age))
mean_img=np.sum(image_matrix,axis=0)/len(image_matrix) # mean photo
mean_photo=np.reshape(mean_img,(360,480),'C')
name="average_photo.jpg"
cv2.imwrite(os.path.join(folder,name),mean_photo)
name1="face1.jpg"
name2="face2.jpg"
name3="face3.jpg"
name4="face4.jpg"
slika1=np.reshape(VT[0,:].T,(360,480))
slika2=np.reshape(VT[1,:].T,(360,480))
slika3=np.reshape(VT[2,:].T,(360,480))
slika4=np.reshape(VT[3,:].T,(360,480))
ret,slika11=cv2.threshold(slika1,0,175,cv2.THRESH_BINARY)
ret,slika22=cv2.threshold(slika2,0,175,cv2.THRESH_BINARY)
ret,slika33=cv2.threshold(slika3,0,175,cv2.THRESH_BINARY)
ret,slika44=cv2.threshold(slika4,0,175,cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(folder_jedinicna_lica,name1),slika11)
cv2.imwrite(os.path.join(folder_jedinicna_lica,name2),slika22)
cv2.imwrite(os.path.join(folder_jedinicna_lica,name3),slika33)
cv2.imwrite(os.path.join(folder_jedinicna_lica,name4),slika44)
niz_sigma=list()
broj_sigma=0
broj_sigma_niz=list()
for x in Sigma:
    niz_sigma.append(x)
    broj_sigma_niz.append(broj_sigma)
    broj_sigma=broj_sigma+1
niz_sigma=np.array(niz_sigma)
broj_sigma_niz=np.array(broj_sigma_niz)
plt.plot(broj_sigma_niz,niz_sigma,marker='o',linewidth=2,markersize=12)
plt.xlabel('i')
plt.ylabel('sigma_i')
plt.title('Sopstvene vrednosti')
plt.show()