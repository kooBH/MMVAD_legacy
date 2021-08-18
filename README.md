# Multi Modal Voice Activity Detection   

## To load face_alginment  
```bash
git submodule init
git submodule update

```  


## input
audio = (n_batch, n_frame, n_mels) = (B, 50, 321)        
  - 10ms shift, 321-mel filterbank    
 
video = (n_batch, n_frame, grayscale, width, height) = (B, 30, 1, 224,224)    
  - 60 fps, cropped box of face   
  
## output 
VAD labels for each video frame = (B, 15)  

## pretrained model data    
https://drive.google.com/file/d/1t_rGo-GCUApnkrlUb618_n1AzbfpRZ2U/view?usp=sharing  

## NOTE  

see ```train.py``` for detailed usage.  
