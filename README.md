# Mask-Detector
[2020. 09. - 2021. 05.] 졸업프로젝트 - 마스크 착용 여부를 탐지하는 자동 비행 드론
---
### 0. 프로젝트 개요
<img width="500" alt="image" src="https://user-images.githubusercontent.com/96675556/163590307-b4244017-d560-41df-b488-5c78276edf43.png">

드론을 이용하여 마스크 미착용자를 감지하고, 미착용자에 대한 위치 정보를 관계자에게 전송하는 것을 목표로 합니다.  
구현에 있어서는 스포츠 경기장과 같이 탐지 범위 내에 있는 사람들이 동일한 방향을 보고 있음과 드론이 정해진 구역을 자동 비행함을 가정합니다.  
본 프로젝트의 목표 수행 과정은 아래와 같습니다.

- 드론은 정해진 포인트 마다 호버링 하며 자동 비행한다.
- 호버링 시 해당 구역의 영상을 촬영하고 마스크 착용 여부를 판단한다.
- 다음 포인트로 이동하며 인식 결과 이미지를 전송한다.

**본 레포지토리는 전체 과정 중 탐지 모듈을 위한 모델 학습에 대한 내용을 담고있습니다.**  

---

### 1. 탐지 알고리즘 

<img width="450" alt="image" src="https://user-images.githubusercontent.com/96675556/163592619-8f9c1fea-559b-431d-a4a5-538666d37b79.png">

### 2. 데이터셋

<img width="450" alt="image" src="https://user-images.githubusercontent.com/96675556/163593379-0bee92ed-c7c4-44ef-b401-618665059a57.png">

#### train set
- with mask : 2250장의 이미지, [인공지능이 만든 사람 이미지](https://generated.photos/)에 마스크 이미지를 합성하여 사용  
- without mask : 2250장의 이미지, 인공지능이 만든 이미지 사용, 마스크를 올바르게 착용하지 않은 경우도 포함

#### test set
사용 환경을 고려하여 실제 사람 이미지로 구성,  
학습 데이터와 마찬가지로 마스크를 올바르게 착용하지 않은 경우도 [without mask] 클래스에 포함

### 3. 학습 모델

<img width="450" alt="image" src="https://user-images.githubusercontent.com/96675556/163593685-2f861d0e-f86d-41e7-b141-6f0bd313b6ca.png">

임베디드 환경에서 사용됨을 고려해 모델의 크기가 작고 파라미터수가 적은 MobileNetV2를 이용한 전이학습 진행

### 4. 학습 결과

![acc_plot](https://user-images.githubusercontent.com/96675556/163594214-49612138-85f2-4e2f-a5cf-fe487c15d145.png)
![loss_plot](https://user-images.githubusercontent.com/96675556/163594222-7bdc5303-3d24-4f6e-ba6a-c81d4842d3ca.png)

<마스크를 올바르게 착용하지 않은 경우>
<img width="450" alt="image" src="https://user-images.githubusercontent.com/96675556/163594550-21e18f56-090c-432b-bcc3-868a2a25deba.png">
<마스크를 올바르게 착용한 경우>
<img width="450" alt="image" src="https://user-images.githubusercontent.com/96675556/163594689-1b603aa4-0ed8-40b5-991c-0328da71d074.png">

