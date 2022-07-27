# intent-classifier
Это репозиторий для нейросети - классификатора команд, которая входящий текст относит к той или иной задаче/команде, которую голосовой помощник должен выполнить.

![Scheme idea](https://github.com/GDSC-Saratov/va-intent-classifier/blob/research/README/scheme_idea.jpg)
## second_model (research)
Было принято решение написать кастомную модель на основе того же [transformers/models/bert](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert) с использованием библиотеки **PyTorch** и предобученной модели **BERT Multilingual Base Uncased**

![Scheme second_model layers](https://github.com/GDSC-Saratov/va-intent-classifier/blob/research/README/scheme_second_model_layers.jpg)
### Technical
- [x] Initial version of model class
- [x] Preparing dataset for training model
- [x] Initial version of model train
  - [x] Boost with GPU
- [x] Model save, load and predict -> `model.save()` already exists
### Scientific
- [x] Define classification layers
- [ ] Define optim func in `model.train()`
- [ ] Prepare dataset
  - [x] For debug (*small_dataset*)
  - [ ] For release
### General
- [ ] Train and save model
- [ ] Write interface (on **Flask**)
