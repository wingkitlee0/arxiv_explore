import numpy as np
import json
import boto3
import uuid
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json, model_from_config
from simpletokenizer import SimpleTokenizer

config_file = "config.json"

class AstrophPrediction:
    """Main class for loading the deep learning model and make predictions
    """
    def __init__(self, config_json=config_file):
        """
        
        """
        self.load_config(config_json)
        print("# Done loading configuration")

        # download the files
        self.filepaths = self.get_s3_files()
        print("# Done downloading the model files")

        self.tokenizer = SimpleTokenizer(self.filepaths['dictionary'], max_words=self.max_words)

        with open(self.filepaths['model'], 'r') as json_file:
            architecture = json_file.read()
            self.model = model_from_json(architecture)

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        self.model.load_weights(self.filepaths['model_weights'], by_name=False)
        print("# Done loading the model and its weights.")


    def load_config(self, config_json):
        """
        load the config json file
        """
        self.config_json = config_json
        with open(self.config_json, "r") as jf:
            self.config = json.load(jf)

        self.maxlen = self.config['maxlen']
        self.max_words = self.config['maxwords']
        # target names
        self.target_name_dict = self.config['target_name_dict']
        self.target_name = [k for k, v in self.target_name_dict.items()]
        self.target_fullname_dict = self.config['target_fullname_dict']
        self.target_fullname = [k for k, v in self.target_fullname_dict.items()]
        self.s3_bucket_name = self.config['s3_bucket_name']


        # file names
        self.keys = ['dictionary', 'model', 'model_weights']
        self.dictionary_name = self.config['dictionary']
        self.model_name = self.config['model']
        self.model_weights_name = self.config['model_weights']
        self.filedict = {'dictionary' : self.dictionary_name,
                        'model' : self.model_name,
                        'model_weights' : self.model_weights_name}


    def get_s3_files(self):
        """
        get the files from s3 bucket if the files do not exist
        """
        s3_client = boto3.client('s3')
        #download 
        filepaths = {}


        download_dir = "/tmp"
        for key, filename in self.filedict.items():
            file_found = False
            for f in os.listdir(download_dir):
                if f.endswith(filename):
                    filepaths[key] = os.path.join(download_dir, f)
                    file_found = True

            if not file_found:
                print("# downloading {}".format(filename))
                download_path = '/tmp/{}{}'.format(uuid.uuid4(), filename)
                s3_client.download_file(self.s3_bucket_name, filename, download_path)
                filepaths[key] = download_path      

        return filepaths      

    def predict(self, texts):
        """routine to make prediction on texts

        Args:
            texts: a list of strings
        """
        seq = pad_sequences(
                self.tokenizer.texts_to_sequences(texts),
                maxlen=self.maxlen)
        return self.model.predict_proba(seq)



if __name__=='__main__':
    prediction = AstrophPrediction()

    texts = ["In this paper, we analyze a suite of isolated galaxy simulations. We find that spiral density wave theory are correct. In particular, it correctly predict the growth of two-armed spiral structure. The star formation are triggered by the spiral waves. The pattern speed is consistent with the observation of corotation in the galaxy sample.",
        "We discovered a new forming planet. This planet has ten Jupiter-mass and is embedded in a protoplanetary disks.",
         "We show that the mass fraction of GMC gas (n>100 cm^-3) in dense (n>>10^4 cm^-3) star-forming clumps, observable in dense molecular tracers (L_HCN/L_CO(1-0)), is a sensitive probe of the strength and mechanism(s) of stellar feedback. Using high-resolution galaxy-scale simulations with pc-scale resolution and explicit models for feedback from radiation pressure, photoionization heating, stellar winds, and supernovae (SNe), we make",
         "We have built a new telescope.",
         "We have observed a new sun spot.",
         "We found that Pluto is indeed a Planet.",
         "We found a new neutron star. This neutron star has a very strong magnetic field.",
         "We discovered the B-modes in the cosmological microwave background, which are the imprints of the primodal density fluctuation. This has a great impact on the understanding of cosmology and inflation.",
         "Enter your text here"
        ]

    results = prediction.predict(texts)
    
    print(prediction.target_name)
    for p in results:
        print(p)
        print(prediction.target_name[np.argmax(p)])
