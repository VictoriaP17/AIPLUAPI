from flask import Flask,render_template,request
import requests
import json
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import io

app=Flask(__name__)
#app.template_folder = ''   #CHANGE TO TEMPLATES FOLDER IF NEEDED

class ExistingProductsInfo:
    def __init__(self,products_data,product_images_ids):
        self.products_data=products_data
        self.product_images_ids=product_images_ids

class WebParameters:
    def __init__(self,session,ws_domain,base_input_url):
        self.session=session
        self.ws_domain=ws_domain
        self.base_input_url=base_input_url
        

servicesHeader={"Content-Type":"application/json"}

def login(webParameters):
    loginURL= webParameters.ws_domain+'login.jsp'
    loginData={'username':'','password':''}     #CHANGE TO USERNAME AND PASSWORD
    loginResponse=webParameters.session.post(loginURL,headers=servicesHeader,json=loginData)

    return loginResponse

def getExistingProductsIDs(productsResponse):
    
    product_images_ids=[]

    product_info_dict=json.loads(productsResponse)

    products_data=product_info_dict["data"]
    for product in products_data:
        if product.get("picture") is not None:
            product_image_id=product["picture"]["id"]
            product_images_ids.append(product_image_id)

    existingProducts = ExistingProductsInfo(products_data,product_images_ids)
    return existingProducts

def createEstimator(data,labels):
    x_train_flat, x_test_flat, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
        
    x_train = np.array([img.flatten() for img in x_train_flat])
    x_test = np.array([img.flatten() for img in x_test_flat])

    classifier = SVC()

    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
    grid_search = GridSearchCV(classifier, parameters)
    grid_search.fit(x_train, y_train)
    best_estimator = grid_search.best_estimator_

    return best_estimator

def validateImageType(img_type_response):
    metaFile_info_dict=json.loads(img_type_response)
    metaFile_data=metaFile_info_dict["data"]  
    metaFile_data_dict=metaFile_data[0]

    return metaFile_data_dict["fileType"]   

def predictImage(loaded_classifier,searched_img):
    searched_img_data=[]
    searched_img=resize(searched_img,(15,15))
    searched_img_data.append(searched_img.flatten())
    searched_img_data=np.asarray(searched_img_data)
    y_pred=loaded_classifier.predict(searched_img_data)

    predicted_image_id=y_pred[0]

    return predicted_image_id

def getPredictedProductInfo(existingProducts,predicted_image_id,predicted_product_dict):
    for product in existingProducts.products_data:
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

                break

    return predicted_product_dict


@app.route('/webservices/json/', methods=['GET','POST'])
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

        base_input_url=base_metafile_url+requested_img_id
        input_url=base_metafile_url+requested_img_id+base_metafile_endpoint

        session = requests.Session()

        webParameters = WebParameters(session,ws_domain,base_input_url)

        loginResponse = login(webParameters)

        if loginResponse.status_code!=200:
            webParameters.session.close()
            return {"error":"Could not login to WS, was login information changed?"},400,servicesHeader


        productsURL= webParameters.ws_domain+'ws/rest/com.axelor.apps.base.db.Product'
        productsResponse=webParameters.session.get(productsURL)

        if productsResponse.status_code!=200:
            return {"error":"Could not get products info"},400,servicesHeader
    
        existingProducts = getExistingProductsIDs(productsResponse.text)
        

        data=np.array([],dtype=object)
        labels=[]

        for image_id in existingProducts.product_images_ids:
            img_response=webParameters.session.get(base_metafile_url+str(image_id)+base_metafile_endpoint)
            
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

        best_estimator = createEstimator(data,labels)
    
        pickle.dump(best_estimator, open('./model.p','wb'))

        with open ('./model.p','rb') as f:
            loaded_classifier = pickle.load(f)

        
        img_type_response=webParameters.session.get(webParameters.base_input_url)

        if img_type_response.status_code!=200:
            return {"error":"Could not fetch uploaded image data: 1"},400,servicesHeader

        allowed_img_formats=("image/jpeg","image/png")
        img_filetype = validateImageType(img_type_response.text)

        if img_filetype not in allowed_img_formats:
            return {"error":"Uploaded file was most likely not an image, please, uploade a valid image file: 1"},400,servicesHeader

        #Reading uploaded image
        searched_img_response=webParameters.session.get(input_url)
        
        if searched_img_response.status_code!=200:
            return {"error":"Could not fetch uploaded image data: 2"},400,servicesHeader

        try:
            searched_img_response_data=io.BytesIO(searched_img_response.content)
            searched_img=imread(searched_img_response_data)
        except:
            return {"error":"Uploaded file was most likely not an image, please, uploade a valid image file: 2"},400,servicesHeader
        

        predicted_image_id = predictImage(loaded_classifier,searched_img)

        predicted_product_dict={
                                "status":0,
                                "offset":0,
                                "data":{}
                                }


        predicted_product_dict = getPredictedProductInfo(existingProducts,predicted_image_id,predicted_product_dict)

        json_response=json.dumps(predicted_product_dict,indent=4)

        #logout and close session
        logoutURL=ws_domain+'logout'
        webParameters.session.get(logoutURL)
        webParameters.session.close()

        return json_response,200,servicesHeader
        

if __name__ =="__main__":
    app.run()
