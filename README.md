1. generate channel model for training and testing:
   - Statics model will use file ChannelData_for_ResCNN.m
   - Mixed model will use file ChannelData_for_ResCNN_mixed.m, modify model range based on multipath value
2. training model
   - using file (for_window)channel_denoise_ResCNN.py will take input as channel, and output as deviation between input and output training
   - using file (for_window)channel_denoise_ResCNN_direct_as_output.py will take input as channel, output will be training output
3. testing model to see error deviation


### improve code for better training ###

tệp training cải thiện được lưu trong folde ./adjustablecode, quá trình chạy code thực hiện như sau:
TRAINING:
 - tạo mẫu kênh truyền bằng file matlab ChannelData_for_ResCNN_mixed.m
   + L_f, L_n sẽ là số đa đường max farfield và nearfiel đạt được
   + kênh khi tạo sẽ được generate random trong khoảng 0 cho đến số đường truyền maximum
   + Để kiểm soát số mẫu được gen ra, thay đổi tham số "num_Channel", num_Channel sẽ là tổng số mẫu được gen ra và phải chia hết cho num_sta*num_ffading
 - Khi training, sử dụng file channel_denoise_ResCNN_adjust_SNRR.py:
   + Thay đổi dải SNR training bằng cách điều chỉnh các tham số
      "snr_min=-10
       snr_max=20
       snr_increment=5
       snr_count = int((snr_max-snr_min)/snr_increment)"
 - Sau khi train sẽ gen ra một file log lưu trữ loss thay đổi, và 1 file output testing nmse ở file train
TESTING:
 - testing sẽ chỉ gen ra mẫu kênh truyền sử dụng 1 tham số đa đường cho Lf và Ln (giống với code gốc), thực hiện gen thông qua file "ChannelData_for_ResCNN.m"
 - Khi testing, sử dụng file "channel_denoise_ResCNN_test.py", thay đổi SNR giống với code training, output sẽ được save vào file csv nmseSummary.csv

