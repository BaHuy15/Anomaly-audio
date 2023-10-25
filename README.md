# Anomaly-audio
## Table of Contents

- [Table of Contents](#table-of-contents)
- [Install Requirements](#Install-Requirements)
- [Data Format](#Data-Format)
- [Generate auto-augmented data](#Generate-augmentation-data)
- [Download Yolov7 Weights](#Download-Yolov7-Weights)
- [Evaluation](#Evaluation )
- [Training](#Training)
- [Result](#Result)
- [Citation](#Citation)
- [Acknowledgement](#Acknowledgements)

## Data Format
<details><summary> <b>Expand</b> </summary> 

``` shell              
Your dataset                                 
 |                                
 |                               
 |______Normal                                                      
 |        |______train                           
 |                 |________normal_data.wav                                         
 |_______Abnormal                      
 |        |______test                     
 |                 |__________anomaly.wav                                    

```                                             
</details>  