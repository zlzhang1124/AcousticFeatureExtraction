# Python code for acoustic feature extraction using Librosa library and openSMILE toolkit.

Simple acoustic feature extraction using the Librosa audio processing library and the openSMILE toolkit, including prosodic features (duration, short-term energy, zero crossing rate, fundamental frequency, etc.), spectral related features (MFCC features), and sound quality features (formant, jitter, shimmer).

If it's of any use to you, please  click a **"Star"**  ~).

[![GPL-3.0 Licensed](https://img.shields.io/badge/License-GPL3.0-blue.svg?style=flat)](https://opensource.org/licenses/GPL-3.0) [![Python Version](https://img.shields.io/badge/Python-3.x-orange.svg)](https://www.python.org/) [![openSMILE Version](https://img.shields.io/badge/openSMILE-2.3.0-brightgreen.svg)](https://www.audeering.com/opensmile/) [![Librosa Version](https://img.shields.io/badge/Librosa-0.7.2-green.svg)](https://github.com/librosa/librosa) 

**ReadMe Language** | [中文版](https://github.com/Zhangtingyuxuan/AcousticFeatureExtraction/README.md) | English |

## Introduction

​    Before this, I know nothing about acoustics. However, I read some literature and blogs to meet the needs of laboratory projects. I just entered the door.

​    This code was developed based on python3.6, running on Windows10, other versions have not been tested for the time being, but I feel that python3.x, the ubuntu platform should be able to run normally. If not, according to the error slightly modified adaptation. I try to make comments on every line in the code.

**Note** : Because I also learn while writing this code, many professional acoustics term I am also a little knowledge, no in-depth study. According to what I can understand, plus some of my own understanding, this code ( perhaps it is better to call the demo), it is inevitable that the extracted features will be inaccurate, especially in the fundamental frequency tracking and formant estimation, the methods are different, and the accuracy varies greatly. You can try to change the parameters, use the latest algorithm, or directly use the features extracted by the openSMILE feature set in this code to avoid repetitive wheel creation~ 

* **About the code**

​    This code uses two methods to extract acoustic features, one is to directly call the feature configuration file in openSMILE, including the 2016-eGeMAPS feature set, a total of 88 features; 2016-ComParE feature set, a total of 6373 features; 2009- InterSpeech Emotion Challenge feature set (IS09_emotion), a total of 384 features. These feature sets are mainly used for speech-based emotion recognition and can also be used for speech recognition. Relevant literature about their detailed introduction:

* [Eyben, Florian, et al. "The Geneva minimalistic acoustic parameter set (GeMAPS) for voice research and affective computing." *IEEE transactions on affective computing* 7.2 (2015): 190-202.](https://ieeexplore.ieee.org/iel7/5165369/7479593/07160715.pdf)

* [Weninger, Felix, et al. "On the acoustics of emotion in audio: what speech, music, and sound have in common." *Frontiers in psychology* 4 (2013): 292.](https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00292)

* [Schuller, Björn, et al. "The INTERSPEECH 2013 computational paralinguistics challenge: Social signals, conflict, emotion, autism." *Proceedings INTERSPEECH 2013, 14th Annual Conference of the International Speech Communication Association, Lyon, France*. 2013.](https://mediatum.ub.tum.de/doc/1189705/file.pdf)

* [Schuller, Björn, et al. "The interspeech 2016 computational paralinguistics challenge: Deception, sincerity & native language." *17TH ANNUAL CONFERENCE OF THE INTERNATIONAL SPEECH COMMUNICATION ASSOCIATION (INTERSPEECH 2016), VOLS 1-5*. 2016.](http://livrepository.liverpool.ac.uk/3003234/1/is2016_compare.pdf)

* [Schuller, Björn, Stefan Steidl, and Anton Batliner. "The interspeech 2009 emotion challenge." *Tenth Annual Conference of the International Speech Communication Association*. 2009.](https://mediatum.ub.tum.de/doc/980035/file.pdf)

  ​    For the introduction of the openSMILE toolkit and its configuration files, please refer to the official documentation [openSMILE book] (https://www.audeering.com/download/opensmile-book-latest/). This program already contains the compiled version of the toolkit and its required configuration files, which are located in the ./openSMILE folder, so there is no need to download and compile again.

  ​    The second is to use the Librosa audio processing library to extract some commonly used acoustic features, including prosodic features (voiced duration, soft duration, effective speech duration, short-term energy, zero-crossing rate, fundamental frequency, log energy and sound pressure level LLD features based on frames, and the HSF features based on the global statistical values of these LLDs for the entire segment of speech, including minimum, maximum, range, mean, standard deviation, skewness, and kurtosis. At the same time, the same operation is performed on the delta and delta-delta), spectrum-based correlation features (39-dimensional MFCC features) and sound quality features (1/2/3 formant center frequency and its bandwidth, jitter, shimmer).

  ​    In addition, the code also includes some speech preprocessing parts: pre-emphasis, framing, windowing, FFT; voice activity detection based on double threshold method; extraction of spectrogram.

  ​    Finally, almost every feature is given a plot, after all, the visualization is really fragrant ~

* **Steps and some details**

​    First clone this code to your computer via Git.
```shell
git clone https://github.com/Zhangtingyuxuan/AcousticFeatureExtraction.git
```

​    Either download the compressed package directly, or you can use the **"Fork"** button to copy a copy, and then clone it locally with your own SSH key.

​    Before running this code, please install the necessary [Python3 version dependences](https://github.com/Zhangtingyuxuan/AcousticFeatureExtraction#python-import)。

​    Run this code directly after cd to the current program path:
```shell
python3 acoustic_feature.py
```
​    There are two audio files under this program./audios path: "audio_raw.wav" (Chinese: 蓝天 白云) and "ae.wav" (English monophthong: [æ]). Running the code based on these two audio files, you will get the following files and graphics output, I will introduce them one by one:

1. The first is the endpoint detection of voice: the output graph is shown in Figure 1/2 below. This can detect all valid speech parts of a section of speech, mainly used for speech pre-processing, and can also be used to achieve speech segmentation based on endpoint detection. More detailed procedures about this method are in my other repositories:[voice_activity_detection](https://github.com/Zhangtingyuxuan/voice_activity_detection)。

   <center>
       <img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/Figure_C1.png" width=90% height=90% />
       <center>Fig1. Chinese：voice activity detection of “蓝天 白云”</center>
   </center>

   <center>
       <img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/Figure_E1.png" width=90% height=90% />
       <center>Fig2. English monophthong：voice activity detection of [æ]</center>
   </center>


   ​    At the same time, the * _vad.wav file after the detection of the first and last endpoints of the corresponding voice file will be generated under the ./audios path, and the next feature extraction is performed through this file. In the ./features folder, a feature.csv feature file in the ARFF format using the feature set corresponding to the openSMILE toolkit will be generated. The content of the file is shown in Figure 3 below.

   <center>
       <img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/Figure.png" width=60% height=60% />
       <center>Fig3. Features extracted from IS09_emotion feature set in openSMILE toolkit</center>
   </center>

2. Prosody feature extraction, comparing the difference between the fundamental frequency(F0), sound pressure level and spectral features of this code and using Praat software(Figure 4/5):

   <center class="half">
       <img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/Figure_C2.png" width=50% height=50% /><img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/20200511113948.png" width=50% height=50% />
       <center>Fig4. Chinese：prosody feature of “蓝天 白云”(left) and using Praat(right)</center>
   </center>

   <center class="half">
       <img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/Figure_E2.png" width=50% height=50% /><img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/20200511115013.png" width=50% height=50% />
       <center>Fig5. English monophthong：prosody feature of [æ](left) and using Praat(right)</center>
   </center>

3. Correlation features based on the spectrum: 39-dimensional MFCC features, including MFCC1-13, where MFCC1 is replaced by logarithmic energy, and then the delta and delta-delta are calculated in sequence (Figure 6).

   <center class="half">
       <img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/Figure_C3.png" width=50% height=50% /><img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/Figure_E3.png" width=50% height=50% />
       <center>Fig6. The visualization of 39-dim MFCC features of Chinese：“蓝天 白云”(left) and English monophthong [æ](right)</center>
   </center>

4. Voice quality features: compared the program with using Praat software in the center of the formant frequency F1/F2/F3 differences (Figure 7/8) :

   <center class="half">
       <img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/Figure_C4.png" width=50% height=50% /><img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/1.png" width=50% height=50% />
       <center>Fig7. Chinese：F1-3 of “蓝天 白云”(left) and using Praat(right)</center>
   </center>

   <center class="half">
       <img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/Figure_E4.png" width=50% height=50% /><img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/2.png" width=50% height=50% />
       <center>Fig8. English：F1-3 of [æ] (left) and using Praat(right)</center>
   </center>

5. Acoustic spectrum: including amplitude spectrogram, power spectrogram, log power spectrogram and log-Mel spectrogram, Figure 9.

   <center class="half">
       <img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/Figure_C5.png" width=50% height=50% /><img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/Figure_E5.png" width=50% height=50% />
       <center>Fig9. Kinds of spectrogram in Chinese：“蓝天 白云”(left) and English monophthong of [æ] (right)</center>
   </center>

## Python Import

Regarding the dependencies of this code (Librosa should be the same as the version I use, other versions have not been tested):

* Librosa-0.7.2
* Numpy-1.18.1
* matplotlib-3.1.3
* Scipy-1.4.1

Special thanks: the development and maintenance personnel of openSMILE and Librosa, the scientific research personnel of acoustics and other related majors, and the dedication and hard work of the big bloggers! Thank you for letting me learn a lot of relevant knowledge!

## License 

[GPL v3.0](LICENSE) © [ZZL](https://github.com/Zhangtingyuxuan)

## Donate

If you like this code and it helps you a little, you can donate here ~

<center class="half">
    <img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/ef76a3d6b636a87f05a769e08910d93.jpg" width=20% height=20% /></div><img src="https://cdn.jsdelivr.net/gh/Zhangtingyuxuan/MyPics@master/img/AliPay.jpg" width=20% height=20% />
    <center>WeChat　　　　　　　AliPay</center>
</center>