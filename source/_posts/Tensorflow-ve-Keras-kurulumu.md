---
title: Tensorflow ve Keras kurulumu
date: 2017-08-03 23:32:21
tags:
---
Bu yazıda Tensorflow ve Keras'ı benim kullandığım şekilde kurmayı anlatacağım. Her ne kadar hem Tensorflow hem de Keras OSX ve Windows destekliyor olsa da hayatın kısa ve acımasız olduğunu göz önüne alarak Virtual Machine bile olsa Ubuntu veya türevi (Lubuntu, Xubuntu vs.) bir sistemde çalışmanızı öneririm.

Aşağıda Tensorflow'un orjinal kurulum dökümanlarının linklerini verdim. Ben aşağıda kendi kullandığım gibi, Ubuntu üzerinde, virtual environment ile ve tüm hesaplamaları CPU'da yapacak şekilde kurulumunu anlatacağım. Minimum 4-5 senelik bir CPUya da bir taşın sabrına sahipseniz burada yayınlayacağım tüm deneyleri bu kurulumla makul sürelerde eğitebilirsiniz.

[Tensorflow Ubuntu Kurulum](https://www.tensorflow.org/install/install_linux "Installing TensorFlow on Ubuntu")
[Tensorflow OSX Kurulum](https://www.tensorflow.org/install/install_mac "Installing TensorFlow on Mac OS X")
[Tensorflow Windows Kurulum](Installing TensorFlow on Windows "Installing TensorFlow on Windows")

# Tensorflow

Bu şekilde yaptığımız kurulumda Python'un paket yöneticisi olan pip ile kurduğumuz tüm paketler bir *virtual environment* içinde kalacak. Aynı makinada başka Python projeleriyle de ilgileniyorsaniz olası versiyon çakışmaları gibi şeyleri engellmesi açısından faydalı.

Burada *virtual environment*'i `~/tensorflow` dizinine kurduğumuzu varsaydım. Ama başka bir yere kurmak istiyorsanız komutlardaki dizini değiştirebilirsiniz.

1. Önce Python'u ve virtual environment yaratmak için ihtiyacımız olan paketleri kuralım.

  `$ sudo apt-get install python-pip python-dev python-virtualenv`

2. Şimdi içine kurulum yapacağımız *virtual environment*'i yaratalım.

  `$ virtualenv --system-site-packages ~/tensorflow`

3. Yarattığımız *virtual environment*'a geçiyoruz.

  `$ source ~/tensorflow/bin/activate`

  Bu işlemi tensorflow'u kullanmak istediğiniz tüm terminallerde yapmanız gerekli. Bu komutu çalıştırınca aşağıdaki bash prompt'unuz aşağıdaki gibi gözükecek. Bu bize tensorflow environment'ının içinde olduğumuzu gösteriyor.

  `(tensorflow)$ `

4. Öncelikle elimizdeki muhtemelen eski olan pip versiyonunu güncelleyelim:

  `(tensorflow)$ pip install --upgrade pip`

5. Şimdi pip kullanarak tensorflow'u kuruyoruz.

  `(tensorflow)$ pip install --upgrade tensorflow`

  Kurulum dökümanı bu komutun bazen patladığını söylüyor. Eğer sizde de çalışmazsa aşağıdaki komutu deneyin:

  ```
  (tensorflow)$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp27-none-linux_x86_64.whl
  ```
6. Machine learning'çilerin favorisi Jupyter Notebook'u da kuralım.

  `(tensorflow)$ pip install --upgrade jupyter`

Artık Jupyter Notebook içinde veya direkt olarak Python'da Tensorflow kütüphanesini kullanabilir haldeyiz.

# Keras

Tensorflow harika bir machine learning kütüphanesi. Tensorflow'u kullanarak okuduğunuz (ve benim aksime matematiğini anlamayı başardığınız herhangi bir *state-of-the-art* machine learning makalesini implemente edebilirsiniz. Ama ne yazık ki benim gibi bu işe yeni başlayanlar ve matematik altyapısı zayıf olanlar için biraz fazla low-level kalıyor.

Keras, Tensorflow veya Theano üzerinde çalışabilen daha üst seviye (high level) bir kütüphane. Sık kullanılan Machine Learning katmanları ve *optimizerlar* gibi şeyler zaten Keras'ın içinde tanımlanmış durumda. Bu da işimizi çok kolaylaştıracak. Yine pip ile kurulabiliyor.

`(tensorflow)$ pip install --upgrade keras`

Keras'ı da kurduktan sonra artık kendimize bir çalışma dizini oluşturup, jupyter'i açarak birşeyler yazmaya başlayabiliriz.

```
(tensorflow)$ mkdir notebook && cd notebook
(tensorflow)$ jupyter-notebook
```

Bu komutu çalıştırdığınız sistemin varsayılan tarayıcısında jupyter notebook açılacak.

---

Bir sonraki yazıda ilk neural network'ümüzü eğiteceğiz.
