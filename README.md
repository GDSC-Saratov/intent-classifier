# intent-classifier
Это репозиторий для нейросети - классификатора команд, которая входящий текст относит к той или иной задаче/команде, которую голосовой помощник должен выполнить.
## second_model (research)
В первой модели возникла проблема с предиктом - предполагалось, из-за неподходящей функции активации в пулере. Было принято решение написать кастомную модель с изменненным пулером на основе того же [transformers/models/bert](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert). Однако кол-во возникающих ошибок во время настройки трейнера для нее оказалось больше, чем нервных окончаний. Поэтому было решено написать чистую модель на **PyTorch** на основе архитектуры **BertModelForSequenceClassification** из [transformers/models/bert/modeling_bert.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) и предобученной модели **Multilingual BERT Base Cased** из [google-research/bert](https://github.com/google-research/bert).
### Technical
- [x] Inital version of classes for model architecture
- [x] Loading state dict (weights and biases) in model from pretrained (BERT Multilingual Base)
- [x] Preparing dataset for training model
- [ ] Training model
  - [ ] Boost with GPU
- [ ] Model predict
  - _Disadvantage_ Model returns `torch.tensor` with shape `(512, 6)`, need with shape `(6)` (number of labels)
### Scientific
- [ ] Define loss and activation funcs
- [ ] Prepare dataset
  - [x] Debugging (small_dataset)
  - [ ] Production