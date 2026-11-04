# FAQ
## iosの動作確認バージョンを教えてください
PowerPredictor : iOS 26
PowerPredictor_feature_ext : MacOS 26

で動作確認しています。最新の

PowerPredictor : iOS 27
PowerPredictor_feature_ext : MacOS 27

でも動作確認済みです。

## Android版はリリースされないのか

予定はありません。Vision Framework, MLX SwiftはいずれもApple社の技術であり、Androidに対応させるには他の技術を採用する必要があります。Androidにもそれぞれに対応した技術はあるようですが同じ会社の技術ではないため、互換性などの観点で不利と考えております。しかしながら有志によるAndroid版の開発を歓迎します。

## iPadでのリリースも可能か。

可能です。iPad Pro 11‑inch (第2世代) の実機へビルドし動作確認しています。

## 測定後の画像は保存されるのか

保存されません。測定アプリであること、を重視しています。

## Powerのブレが大きい

・学習量が少ないこと
・DINOv3のCLS Tokenの性質上、同じ人であっても着ている服や姿勢などもPowerに反映されるためブレることがあります。Trainingのコード例、Training方法も公開していますのでぜひご自身での改善を試してください。