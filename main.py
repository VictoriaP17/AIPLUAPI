from flask import Flask,render_template,request
import requests
import json
import pickle
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import io
import imghdr

app=Flask(__name__)
#app.template_folder = ''   #CHANGE TO TEMPLATES FOLDER IF NEEDED

def is_image(filepath):
    image_type=imghdr.what(filepath)
    return image_type is not None


servicesHeader={"Content-Type":"application/json"}

@app.route('/webservices/json', methods=['GET','POST'])
def index():
    if request.method=='GET':
        return render_template('index.html')
    
    if request.method=='POST':

        if not request.form['txtSearchedImageId']:
            return {"error":"No image to look for, please, enter an image ID"},400,servicesHeader

        ws_domain=''  #CHANGE TO WS DOMAIN
        base_metafile_url=ws_domain+'ws/rest/com.axelor.meta.db.MetaFile/'
        base_metafile_endpoint="/content/download"

        requested_img_id=request.form['txtSearchedImageId']

        input_url=base_metafile_url+requested_img_id+base_metafile_endpoint

        session = requests.Session()

        #login to server
        loginURL= ws_domain+'login.jsp'
        loginData={'username':'','password':''}     #CHANGE TO USERNAME AND PASSWORD

        loginResponse=session.post(loginURL,headers=servicesHeader,json=loginData)

        if loginResponse.status_code!=200:
            return {"error":"Could not login to WS, was login information changed?"},400,servicesHeader

        #get the id of all of the product's images
        productsURL= ws_domain+'ws/rest/com.axelor.apps.base.db.Product'

        productsResponse=session.get(productsURL)

        if productsResponse.status_code!=200:
            return {"error":"Could not get products info"},400,servicesHeader
        

        product_images_ids=[]


        product_info_dict=json.loads(productsResponse.text)

        products_data=product_info_dict["data"]
        for product in products_data:
            if product.get("picture") is not None:
                product_image_id=product["picture"]["id"]
                product_images_ids.append(product_image_id)

        data=np.array([],dtype=object)
        labels=[]

        for image_id in product_images_ids:
            print(base_metafile_url+str(image_id)+base_metafile_endpoint)
            img_response=session.get(base_metafile_url+str(image_id)+base_metafile_endpoint)
            
            if img_response.status_code!=200:
                return {"error":"Could not fetch image data"},400,servicesHeader
            
            image_data=io.BytesIO(img_response.content)

            img=imread(image_data)
            img=resize(img,(15,15,3))
            for i in range(10):
                data=np.append(data,img.flatten())
                labels.append(image_id)
            
        data = data.reshape(-1,15,15,3)
        labels=np.asarray(labels)

        
        x_train_flat, x_test_flat, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
        

        x_train = np.array([img.flatten() for img in x_train_flat])
        x_test = np.array([img.flatten() for img in x_test_flat])

        classifier = SVC()

        parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
        grid_search = GridSearchCV(classifier, parameters)
        grid_search.fit(x_train, y_train)
        best_estimator = grid_search.best_estimator_
        y_prediction = best_estimator.predict(x_test)
        score = accuracy_score(y_prediction, y_test)
        print('{}% of samples were correctly classified'.format(score * 100))
        pickle.dump(best_estimator, open('./model.p','wb'))

        with open ('./model.p','rb') as f:
            loaded_classifier = pickle.load(f)

        searched_img_data=[]
        print(input_url)
        searched_img_response=session.get(input_url)
        
        if searched_img_response.status_code!=200:
            return {"error":"Could not fetch uploaded image data"},400,servicesHeader
        
        try:
            searched_img_response_data=io.BytesIO(searched_img_response.content)
            searched_img=imread(searched_img_response_data)
        except:
            return {"error":"Uploaded file was most likely not an image, please, uploade a valid image file"},400,servicesHeader
        #searched_img=imread(input_dir)

        searched_img=resize(searched_img,(15,15))

        searched_img_data.append(searched_img.flatten())
        
        searched_img_data=np.asarray(searched_img_data)

        y_pred=loaded_classifier.predict(searched_img_data)

        predicted_image_id=y_pred[0]

        predicted_product_dict={
                                "status":0,
                                "offset":0,
                                "data":{}
                                }

        for product in products_data:
            if product.get("picture") is not None:
                current_image_id=product["picture"]["id"]
                if predicted_image_id==current_image_id:
                    predicted_product_dict['data']['id']=product.get('id')
                    predicted_product_dict['data']['name']=product.get('name')
                    predicted_product_dict['data']['code']=product.get('code')
                    if product.get("unit") is not None:
                        predicted_product_dict['data']['unit']=product.get('unit')
                    if product.get("salePrice") is not None:
                        predicted_product_dict['data']['salePrice']=product.get('salePrice')

                    #logout of server
                    logoutURL=ws_domain+'logout'
                    session.get(logoutURL)

                    json_response=json.dumps(predicted_product_dict,indent=4)
                    print(json_response)
                    return json_response,200,servicesHeader

        
        #logout of server
        logoutURL=ws_domain+'logout'
        session.get(logoutURL)

        return {"error":"Could not find an appropriate product"},400,servicesHeader

if __name__ =="__main__":
    app.run()
