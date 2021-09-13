# 设置文件位置

# PRE_HANDLE代表处理为字符型的中间文件，NORM代表经过归一化的文件
RAW_TRAIN_PERCENT10 = 'data/10_percent_train.txt'
RAW_TEST = './data/test'
PRE_HANDLE_TRAIN = 'data/pre_train.csv'
PRE_HANDLE_TEST = 'data/pre_test.csv'
NORM_TRAIN = 'data/norm_train.csv'
NORM_TEST = 'data/norm_test.csv'
# 存在csv文件中的norm文件只是为了方便观看，在后面使用pickle存储的numpyArray数据，处理速度快
TRAINDATA = './data/train.pickle'
TESTDATA = './data/test.pickle'

# 设置将离散字符数据转换为离散数值数据的对应字典
protocol_type_list = ['tcp', 'udp', 'icmp']
service_list = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name',
                'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
flag_list = ['OTH', 'REJ', 'RSTO', 'RSTOS0',
             'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']

# 标签数值化的衡量标志，只进行二值化，等于normal就是0，不等于就是1
NORMFLAG = 'normal.'

proto2numDic = {v: k for k, v in enumerate(protocol_type_list)}
service2numDic = {v: k for k, v in enumerate(service_list)}
flag2numDic = {v: k for k, v in enumerate(flag_list)}


# 设置特征名称列表，方便后续任务展示
feature_list = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',  # TCP连接基本特征（共9种） 
                # TCP内容特征(13种)
                'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_hot_login', 'is_guest_login',
                # 基于时间的网络流量特征统计(9种)
                'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                # 基于主机的网络流量特征统计(10种)
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
                ]

num2featureDic = {k: v for k, v in enumerate(feature_list)}


# 只考虑排在前YOU_CHOOSE的特征
YOU_CHOOSE = 10

# KNN中选取最近的K个近邻
K = 5

# 原训练集的采样率
SAMPLE_TRAIN_RATE = .025
# 测试集的采样率
SAMPLE_TEST_RATE = .025

# 存储Fisher运算结果的文件
FISHER_FILE='data/fisher.csv'
