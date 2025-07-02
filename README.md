# Temporal Link Prediction with GraphSAGE

Bu proje, dinamik bir graf üzerinde temporal link prediction görevini gerçekleştirmek için GraphSAGE tabanlı bir model kullanmaktadır.

## 🎯 Proje Amacı

Verilen bir kaynak düğüm, hedef düğüm, kenar tipi ve gelecek zaman penceresi için, bu kenarın belirtilen zaman aralığında oluşma olasılığını tahmin eder.

## 📦 Veri Seti

- `edges_train_A.csv`: Temporal kenarları içerir (src_id, dst_id, edge_type, timestamp)
- `node_features.csv`: Düğüm özelliklerini içerir
- `edge_type_features.csv`: Kenar tipi özelliklerini içerir
- `input_A.csv`: Tahmin yapılacak örnekleri içerir

## 🛠️ Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Veri setlerini proje dizinine yerleştirin:
   - `edges_train_A.csv`
   - `node_features.csv`
   - `edge_type_features.csv`
   - `input_A.csv`

## 🚀 Kullanım

Modeli eğitmek ve tahmin yapmak için:

```bash
python train.py
```

Bu komut:
1. Veriyi yükler ve ön işler
2. GraphSAGE modelini eğitir
3. Tahminleri `output_A.csv` dosyasına kaydeder

## 📊 Model Mimarisi

- **GraphSAGE**: Düğüm komşuluklarından bilgi toplayan ve ölçeklenebilir bir şekilde düğüm gömme vektörleri öğrenen bir graf sinir ağı modeli
- **Temporal Özellikler**: Zaman damgalarını ve kenar tiplerini modelleyen özel katmanlar
- **Link Prediction**: Düğüm gömme vektörlerini kullanarak kenar oluşma olasılığını tahmin eden bir sınıflandırıcı

## 📈 Değerlendirme

Model performansı ROC eğrisi altındaki alan (AUC) metriği ile değerlendirilir.

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 