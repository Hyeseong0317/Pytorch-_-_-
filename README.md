# 캡스톤 디자인 프로젝트(J&K project)

(Kaggle_Dataset)[https://www.kaggle.com/paultimothymooney/kermany2018]

모델관련 파일 or train관련된 파일 발췌

어떤 파라미터를 사용했는지 이런거 보기

사람들이 많이 지정한 평균과 표준편차가 있다. 정규화하는 이유는 모든데이터의 범위자체를 균등하게 분포시켜서 빠르게 학습시킨다.

앞에 있는 layer와 노드를 동일하게 만들어야 오류가 나지 않는다.

모르겠으면 print해봐라

tensor 형식의 이미지들은 4개의 채널이 있다. 1번째가 ~~

넘파이에서는 채널 로우 컬럼->로우 컬럼 채널로 바꿔야한다


### 우리가 지정한 device(GPU0, GPU1, ... 중 하나)로 inputs과 labels를 옮겨 메모리에 저장한다.

inputs = inputs.to(device)

labels = labels.to(device)

### _, preds = torch.max(outputs, 1) 에서 torch.max는 outputs을 2개(labels과, predictions값)출력한다. Labels값은 사용하지 않을 것이므로 _,로 비운다. 이렇게 하는 이유는 outputs이 2개인 torch.max함수의 출력데이터 개수와 형식을 맞춰주어 오류를 방지하기 위해서이다. 

 outputs = model(inputs)
 
            _, preds = torch.max(outputs, 1)

### loss를 구하는 함수를 지정한다. CrossEntropyLoss()가 일반적이다. CrossEntorpyLoss()함수로 지정된 변수 criterion에 인수로 outputs, labels를 주면 모델이 출력한 Outputs과 우리가 아는 labels값을 비교하여 차이(손실, Loss)를 구한다.

loss = criterion(outputs, labels)

criterion = nn.CrossEntropyLoss()


# 모든 매개변수들이 최적화되었는지 관찰

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# 7 에폭마다 0.1씩 학습률 감소

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

### model을 device(GPU)로 옮긴다.

model_ft = model_ft.to(device)



### _, preds = torch.max(outputs, 1) 에서 torch.max는 outputs을 2개(labels과, predictions값)출력한다. Labels값은 사용하지 않을 것이므로 _,로 비운다. 이렇게 하는 이유는 outputs이 2개인 torch.max함수의 출력데이터 개수와 형식을 맞춰주어 오류를 방지하기 위해서이다. 

 outputs = model(inputs)
 
            _, preds = torch.max(outputs, 1)
