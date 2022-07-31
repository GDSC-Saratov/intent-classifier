# intent-classifier
Это репозиторий для нейросети - классификатора команд, которая входящий текст относит к той или иной задаче/команде, которую голосовой помощник должен выполнить.

![Scheme idea](https://github.com/GDSC-Saratov/va-intent-classifier/blob/research/README/scheme_idea.jpg)
## second_model (research)
Было принято решение написать кастомную модель на основе того же [transformers/models/bert](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert) с использованием библиотеки **PyTorch** и предобученной модели **BERT Multilingual Base Uncased**

![Scheme second model layers](https://github.com/GDSC-Saratov/va-intent-classifier/blob/research/README/scheme_second_model_layers.jpg)
### Technical
- [x] Initial version of model class
- [x] Preparing dataset for training model
- [x] Initial version of model train
  - [x] Boost with GPU
- [x] Model save, load and predict -> `model.save()` already exists
### Scientific
- [x] Define classification layers
- [x] Define optim func in `model.train()`
- [ ] Prepare dataset
  - [x] For debug (*small_dataset*)
  - [ ] For release
### General
- [ ] Train and save model
- [ ] Write interface (on **Flask**)

#### Defining optim func in `model.train()`
Было проведено исследование различных конфигураций модели с рекомендуемыми различными исследователями _optimizers_: Adam и AdamW, _learning rate_ для них: 2e-5, 3e-5 и 5e-5 и _epochs_: 2, 3, 4. _batch_size_ для каждой конфигурации выбран общий emdash 16 (график см. ниже).

По моему (@fruitourist) мнению, наиболее подходящим для общей модели оказывается использование в качестве _optimizer_ - **AdamW** из-за ее большей устойчивости по сравнению с Adam (см. на графике), _learning rate_ - **5e-5**. Наиболее подходящий _epochs_ - **4**.

![Plot define optim func](https://github.com/GDSC-Saratov/va-intent-classifier/blob/research/README/plot_define_optim_func.png)

#### Ready model on small_dataset
