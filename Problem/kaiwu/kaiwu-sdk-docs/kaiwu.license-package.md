kaiwu.license package
Module contents
license相关

kaiwu.license.init(user_id, sdk_code)
初始化生成license文件, 每次调用都会重新生成license文件

Parameters
:
user_id – 用户ID

sdk_code – SDK授权码

kaiwu.license.ensure_license()
检查license文件是否存在，如果不存在，在控制台提示用户输入user_id 和 sdk_code来下载license