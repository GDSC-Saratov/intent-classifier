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
- [ ] Define optim func in `model.train()`
- [ ] Prepare dataset
  - [x] For debug (*small_dataset*)
  - [ ] For release
### General
- [ ] Train and save model
- [ ] Write interface (on **Flask**)

#### Defining optim func in `model.train()`
Было проведено исследование различных конфигураций модели с рекомендуемыми [различными исследователями] _функциями оптимизации_ - Adam и AdamW; с различными рекомендуемымыми _learning rate_'ами для них - 2e-5, 3e-5 и 5e-5 и кол-вом _эпох_. batch_size для каждой конфигурации выбран один - 16 (график см. ниже). По моему (@fruitourist) мнению, наиболее подходящей для общей модели оказывается использование в качестве функции оптимизации **AdamW** из-за ее большей устойчивости по сравнению с Adam (см. на графике), learning rate которой равен **5e-5**. Наиболее подходящее кол-во эпох - **3**: именно после данной эпохи нейросеть перестает обучаться лучше - можно предположить, что обучение через большее кол-во эпох может привести к переобучению.

![Plot define optim func](https://github.com/GDSC-Saratov/va-intent-classifier/blob/research/README/plot_define_optim_func.png)