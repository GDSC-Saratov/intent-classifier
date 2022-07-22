# intent-classifier
Это репозиторий для нейросети - классификатора команд, которая входящий текст относит к той или иной задаче/команде, которую голосовой помощник должен выполнить.
## second_model (research)
Было принято решение написать кастомную модель на основе того же [transformers/models/bert](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert) с использованием библиотеки **PyTorch** и предобученной модели **BERT Multilingual Base Cased**
### Technical
- [x] Initial version of model class
- [x] Preparing dataset for training model
- [x] Initial version of model train
  - [x] Boost with GPU
- [x] Model save, load and predict -> `model.save()` already exists
### Scientific
- [ ] Define classification layers
- [ ] Define loss and optim funcs in `model.train()`
- [ ] Prepare dataset
  - [x] For debug (*small_dataset*)
  - [ ] For release
### General
- [ ] Train and save model
- [ ] Write interface (on **Flask**)