import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from json import loads
import requests
import os
import sqlalchemy
from google.cloud import storage
from google.oauth2 import service_account


class Models():

  def __init__(self):

    self.model = load_model(self.download_model())
    self.event = self.get_event_from_sql()

  def summary(self):
    return self.model.summary()

  def predict_to_df(self, user_id: int, ascending=False):
    """
    mendapatkan rekomendasi dalam bentuk dataframe
    user_id : id user
    ascending : urutan data (False: besar=>kecil)|(True: kecil=>besar)
    """
    event = self.event.copy()
    user_ids = np.array([user_id] * len(self.event))
    results = self.model([self.event.id.values, user_ids]).numpy().reshape(-1)

    event['predicted_interest'] = pd.Series(results)
    event = event.sort_values('predicted_interest', ascending=ascending)

    return event

  def predict_to_json(self, user_id: int, ascending=False):
    return loads(self.predict_to_df(user_id, ascending=False).to_json(orient="records"))

  def get_event_from_sql(self):
    """mendapatkan data event dari database dan mereturn kedalam dataframe"""
    try:
        # Replace the placeholders with your MySQL connection details
        host = '34.101.121.14'
        database = 'eventapp'
        user = 'root'
        password = 'root'

        # Membuat URL koneksi database
        url = f'mysql+pymysql://{user}:{password}@{host}/{database}'

        # Membuat objek engine
        engine = sqlalchemy.create_engine(url)

        # Membuat koneksi
        with engine.connect() as connection:
            # Membaca data dari SQL dan mendapatkan DataFrame
            query = sqlalchemy.text('SELECT * FROM event')
            df = pd.read_sql(query, connection)

            # hapus kolom yang  tidak di train pada ml
            return df.drop(['waktu', 'eventimages', 'createdAt', 'updatedAt'], axis=1)

    except sqlalchemy.exc.OperationalError as e:
        print(f'Error connecting to MySQL database: {e}')

  def download_model(self):
    bucket_name = "capstone-eventapp"
    file_name = "recomender/model.h5"
    destination_path = "model/recomender.h5"

    # Membuat objek credentials menggunakan informasi kredensial yang diberikan
    credentials_info = {
        "type": "service_account",
        "project_id": "capstone-387114",
        "private_key_id": "3dea6195f1b82a11c9551b13c98b48d48bc3107e",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDFJxZ1YrAVWx9i\nXXFOoI05a8DZ1R1CkOEDJZ1mSthw6P+j7KFKaevOWDxSdUZtLVXauEkYO4ImAbhg\nd3cLParFQAOz+NA9u24ZA9yUp974SktZxXXeKCYAtnO+HzogH4YR9KaryyxSqg5x\nU6ysXbv3zmKlOdqPMUTXEyiMlFfzrEbFyi1GWi2OrXMij4hvmK/fsbC6YXeqQ5s9\n6WGWGk5Mc/KEdycriPIV10CvwfF9/F7EgRDqwBhA3pa3NhoTluTlE/EtQrZiK+ES\nlp+UKk/OL7mKjk3YyMHcZ8/n8S8WlASwkpHEoocKB4sbL1ZqzsMBpROkpxGvfQVe\n41NFH2TbAgMBAAECggEAWiYQZ+aTW4CKLkFKAUj6qonx6ek/8uMqcHTvrwmERMTV\nuBAIhG1AjN7d3lqPtHZpbpSbn7/+OADLXRjjTzmIb59g0hdwqPmeU0PnxfWox+G4\n6LiYt9el4OeMrx+6RVwEwwsady10++uUpQ88wqgtvhcaFjEJFjbSIoI5JWbjfKC5\nWHV6LUbMxTij00RsmI4vj0I3zPdco4KLRL7nbOtlrwmAeUYpgnFdiK9bflpLEEYn\npGLyjzeIXtcZjtm7JuE7OaucRPcyuM+0evhVvmOn3GqovDm8iN869S4tRiOrBA/g\nbMs8y3LX8mOFZCGgToRkACW7Eqmq4C2/jTR7hfZaaQKBgQDjHxF6XoZXVZ0pszmB\n6EL5/+hk/xRyx9jSBZsbk8SJSSbKYH7DYbxPJJ7eERu8DPCRAr9m4nDjyaDQnSU7\nG7Gj7JjfG5INH+sxwR3EHzT/FOcZjAbhR5V4FRXIvhLT59f498FLm3ECq3s8MNhl\naLLRJqeRGPN4NicQWMbu7+RR1QKBgQDeOIWS31hB3HLIvU2sAGMc7gvIjENgsbsF\nb2enB8lsqQdd6z25dUmF2pU8EaCvxMWDzm3KyKx+vdwrdAS0wxTCTvOgSJcYbw4+\n/h9MNvgDP8ankpibBqF3GlIIW1k7xfNJl1mV29VjjgHq/EmpjA0cMnhaPwzDkLhu\ndYzvqTuD7wKBgAoTB7hw1u6qyoTeAVAE2Gu0cT+BnQuWV8TBIOcxP8eDKihR7W2H\nOU4dZrqc8aj+vfEAuTK8GpvQBsUaI1ui19dYmFNVKr5QSyNy2HoplDU4XSPh9TAq\n97NS5Bt6auVhZFAT1UwgQfwHVTkPRZoB8eSbHVWvUKVlaYgtF+3jUDJdAoGBAMvW\n5Q/4t6iSYugoXZstL71VzpFDuHp2DavKqTXaOrXlxtAr/Q6lu6+A9euH7/HLebKS\nJLwin7gSyGdDoR1+5QfoDCo27AKJvHOj/2mV8qJoWf1Ux49M4cey6RSpVo0x4xza\nLF7+rBsKJFA85AUWZuA6m5OMylSJ5+PLCGC+x80ZAoGAAMEDgIc7zOVx8bENhhTd\nxRD+eAVOOPV8Xv6fIOgQETQfeIseQivqFijb9e/m7nsxnIEf3XFv4F43wU5ZG3gT\nQcZ/Oa07oOfMyU14ZtY5GBEL4PYIt1yr4N6H/PXEKISYIBD87rZ3S2vVqK+eiNDw\nQa04869WwZJ24ilZlKwBjqQ=\n-----END PRIVATE KEY-----\n",
        "client_email": "capstone-387114@appspot.gserviceaccount.com",
        "client_id": "107904801349369921775",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/capstone-387114%40appspot.gserviceaccount.com",
        "universe_domain": "googleapis.com"
    }
    credentials = service_account.Credentials.from_service_account_info(
        credentials_info)

    # Membuat objek client Google Cloud Storage dengan menggunakan kredensial
    client = storage.Client(credentials=credentials)

    # Mengunduh file dari bucket
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.download_to_filename(destination_path)

    return destination_path
