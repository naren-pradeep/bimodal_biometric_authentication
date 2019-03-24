# bimodal_biometric_authentication
To identify or authenticate a person using bimodal biometric system.The biometric features are given as input. Voice is recorded through microphone and
images for palm are obtained using palm vein &amp; vein scanner. 

# Objective
  * To overcome the drawbacks of the unimodal system with more than one modality\
  * To eliminate the False Rejection Rate(FRR) and False Acceptance Rate(FAR)\
  * To authenticate a person accurately using the provided input biometrics\
  * To restrict the access to a system by unauthorized individuals\
  
 # Feature Extraction
  * Speech - MFCC feature are extracted for each samples and stored in a pickle format. The MFCC features are loaded from the pickle file to a numpy array.
   * Palm Image - The palm image features are extracted by 2D Gabor feature extraction method using opencv library

 # Validation
    The extracted features are trained using Support vector machine. 
    
    

    
    
