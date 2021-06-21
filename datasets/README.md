The training data `GiaChungCu_HCM_June2021_laydulieu_com.csv` are taken from http://www.laydulieu.com/nha-dat/ in June 2021. The following columns are filtered:

* Chuyên mục: filter value: "Căn hộ, Chung cư"
* Nhu cầu: filter value: "Cần bán"
* Tỉnh/Thành phố: filter value: "Hồ Chí Minh"

Samples with price <100 triệu đồng (100 million VND) are removed, since there are almost NO apartments in Ho Chi Minh city having those price (they can be mistaken data or noise).


