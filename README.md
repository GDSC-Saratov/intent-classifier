# intent-classifier
Это репозиторий для нейросети - классификатора команд, которая входящий текст относит к той или иной задаче/команде, которую голосовой помощник должен выполнить.
## second_model (research)
Было принято решение написать кастомную модель на основе того же [transformers/models/bert](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert) с использованием библиотеки **PyTorch** и предобученной модели **BERT Multilingual Base Cased**
### Technical
- [x] Inital version of model class
- [x] Preparing dataset for training model
- [x] Training model
  - [x] Boost with GPU
- [x] Model predict
### Scientific
- [ ] Define classification layers
- [ ] Define loss and optim funcs in `model.train()`
- [ ] Prepare dataset
  - [x] For debug (*small_dataset*)
  - [ ] For release
### General
- [ ] Train and save model
- [ ] Write interface (on **Flask**)