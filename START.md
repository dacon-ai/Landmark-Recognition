# Docker Build

모델 테스트를 위해 도커 환경을 빌드합니다. `Dockerfile` 이 있는 루트 디렉토리로 이동하여 아래의 명령어를 실행합니다. 

아래 명령어는 `Dockerfile` 로부터 nia-landmark 라는 image 파일을 생성합니다.

```
sh build_docker.sh
```

Docker 환경이 구성이 되었으면, Docker Image로부터 컨테이너를 생성 한 후 접속합니다.

컨테이너 생성 명령은 `exec_docker.sh` 파일에서 마운트하고자하는 학습용 데이터 경로를 수정 한 후 아래의 명령어를 실행하면 됩니다.


```
sudo docker run -it --name nia -v /your/data/path/data:/Landmark-Recognition/data --gpus all nia-landmark ## 파일 경로를 수정합니다.
```

```
sh exec_docker.sh
```

도커 컨테이너에 접속하게 되면 컨테이너는 아래와 같은 경로를 가지고 있습니다.

Landmark-Recognition
  |------data   
           |------train ## 학습용 데이터셋 경로
           
  |------notebook
           |-----*.ipynb ## 참고용 노트북 파일
           
  |------output          ## 모델 학습 시 저장 될 weight 경로
  
  |------src             ## 소스파일
           |-----dataset
           |-----model
           |-----utils
           |-----train.py
           |-----inference.py  
  |------*
           
           