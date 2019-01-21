import numpy as np
import json

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json, model_from_config
from simpletokenizer import SimpleTokenizer

config_file = "config.json"

"""
maxlen = 100
dictionary_file = "dictionary.json"
model_file = "model.json"
model_weights_file = "model_weights.h5"

target_name_dict = { 'astro-ph.GA' : 0,
                     'astro-ph.SR' : 1,
                     'astro-ph.IM' : 2,
                     'astro-ph.EP' : 3,
                     'astro-ph.HE' : 4,
                     'astro-ph.CO' : 5
                   }
target_name = [k for k, v in target_name_dict.items()]
"""

class AstrophPrediction:
    """Main class for loading the deep learning model and make predictions
    """
    def __init__(self, config_json=config_file):
        """
        
        """
        self.load_config(config_json)

        self.tokenizer = SimpleTokenizer(self.dictionary_file)

        with open(self.model_file, 'r') as json_file:
            architecture = json_file.read()
            self.model = model_from_json(architecture)

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        self.model.load_weights(self.model_weights_file, by_name=True)
        print("# Done loading the model and its weights.")


    def load_config(self, config_json):
        self.config_json = config_json
        with open(self.config_json, "r") as jf:
            self.config = json.load(jf)

        self.maxlen = self.config['maxlen']
        self.dictionary_file = self.config['dictionary_file']
        self.model_file = self.config['model_file']
        self.model_weights_file = self.config['model_weights_file']
        self.target_name_dict = self.config['target_name_dict']
        self.target_name = [k for k, v in self.target_name_dict.items()]

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
         "We discovered the B-modes in the cosmological microwave background, which are the imprints of the primodal density fluctuation. This has a great impact on the understanding of cosmology and inflation."
        ]

    results = prediction.predict(texts)
    
    print(prediction.target_name)
    for p in results:
        print(p)
        print(prediction.target_name[np.argmax(p)])
