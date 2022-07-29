# Tesseract-OCR을 활용한 정확도 높은 한컴오피스 작성문서 인식기

## :fire: 목표 ##

* opencv, tesseract를 활용해 이미지 스캐너를 만들고 글씨를 추출해내기
* 직접 학습시킨 tesseract를 사용해 기존 tesseract보다 더 정확도가 높은 OCR 검출기를 만든다 ##

## :envelope: 프로젝트 설명 ##
* 본 프로젝트는 서울대학교 2021 POLARIS LOC Winter Intership-OCR 인턴 활동 중 진행한 프로젝트 입니다.

* Tesseract-OCR은 Google의 오픈소스 프로젝트로 문자인식 분야에서 광범위하게 사용되고 있습니다. 
  하지만 문서를 작성할 때 널리 쓰이는 한컴오피스 프로그램에서 사용되는 기본 글꼴 30가지를 Tesseract-OCR을 사용하여 인식해 보았을 때
  특정 글꼴에서 인식률이 낮다는 것을 확인할 수 있었습니다. 

* 본 프로젝트는 Tesseract의 미세조정 방식을 사용하여 기존의 인식률이 높게 나왔던 고딕과 명조체뿐만 아니라 정확도가 가장 낮게 나타난 
 ‘가는 안상수체’, ‘양재블럭체’, ‘한컴 쿨재즈B체’, ‘휴먼가는샘체’ 또한 인식이 잘 되는 Tesseract-OCR 모델을 개발하였습니다.
 
* 또한 OpenCv의 이미지 전처리를 통해 문서 영역을 인식한 후 향상된 Tesseract-OCR 모델을 사용해 기존 Tesseract보다 정확도 높은 결과를 도출해냈습니다.
 
## :bulb: 결과 ##

* OCR의 글꼴 인식도 확인
* ![image](https://user-images.githubusercontent.com/80324369/166236425-208a2878-e09c-4c06-abd5-28452aca089b.png)

![image](https://user-images.githubusercontent.com/80324369/166236502-bc9f1ea9-805e-45a5-b1d0-a67efbb28015.png)


* 이미지 전처리 실행
* ![image](https://user-images.githubusercontent.com/80324369/166236534-fe6746c1-b01b-48dd-ae1a-6a8ecef5285a.png)


* OCR 결과
* ![image](https://user-images.githubusercontent.com/80324369/166236715-3a38c8ce-1d56-4b8c-91d0-325715290ffe.png)


## :seedling: 프로젝트 진행 과정 ##
![image](https://user-images.githubusercontent.com/80324369/166236784-1aecdb86-e0b9-46b3-9e79-1c68da8e74a2.png)


