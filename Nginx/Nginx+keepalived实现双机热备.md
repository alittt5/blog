## Nginx+keepalived实现双机热备

```
服务器 两台centos 7
ip 192.168.39.126
ip 192.168.39.120
vip 192.168.39.111
```

### 安装nginx

```
vim /etc/yum.repos.d/nginx.repo
```

添加如下内容：

```
[nginx]

name=nginx repo

baseurl=http://nginx.org/packages/centos/$releasever/$basearch/

gpgcheck=0

enabled=1
```



```
yum install nginx -y
```

 

关闭防火墙

```
systemctl stop firewalld  

 
```

 

关闭主动防御  

```
  临时关闭：
  [root@localhost ~]# getenforce Enforcing     
  [root@localhost ~]# setenforce 0  
  [root@localhost ~]# getenforce  Permissive     
  永久关闭：
  [root@localhost ~]# vim /etc/sysconfig/selinux     SELINUX=enforcing 改为 SELINUX=disabled     重启服务reboot   
```

  

 

修改nginx配置文件

  

```
vim /etc/nginx/conf.d/default.conf
```

```
#
server {
        listen 192.168.39.201:80;
        server_name test.sectoken.io;

        access_log /var/log/nginx/test_access.log;
        error_log /var/log/nginx/test_error.log;

        location / {
            index  index.html index.htm;
            root   /usr/share/nginx/test;
            try_files  $uri $uri/ /index.html;
        }

        error_page 404 /404.html;
            location = /40x.html {
        }

        error_page 500 502 503 504 /50x.html;
            location = /50x.html {
        }
}

server {
        listen 192.168.39.201:80;
        server_name _;
        return 403;
}

```

 

### 安装keepalived

```
yum install keepalived -y
```

 

修改配置文件

```
 vim /etc/keepalived/keepalived.conf
```

  

```
##配置文件
! Configuration File for keepalived

global_defs {
   router_id LVS_01
}

## keepalived 会定时执行脚本并对脚本执行的结果进行分析，动态调整 vrrp_instance 的优先级。
#  如果脚本执行结果为 0，并且 weight 配置的值大于 0，则优先级相应的增加。
#  如果脚本执行结果非 0，并且 weight配置的值小于 0，则优先级相应的减少。
#  其他情况，维持原本配置的优先级，即配置文件中 priority 对应的值。
vrrp_script chk_nginx {
    script "/etc/keepalived/nginx_check.sh"      # 检测 nginx 状态的脚本路径
    interval 2             # 脚本执行间隔，每2s检测一次
    weight -5           # 脚本结果导致的优先级变更，检测失败（脚本返回非0）则优先级 -5
    fall 2                    # 检测连续2次失败才算确定是真失败。会用weight减少优先级（1-255之间）
    rise 1                   # 检测1次成功就算成功。但不修改优先级
}

vrrp_instance VI_1 {
    state MASTER
    interface ens33 #网卡
    virtual_router_id 51
    priority 100
    advert_int 1
    authentication {
        auth_type PASS
        auth_pass 1111
    }

    ## 将 track_script 块加入 instance 配置块
    track_script {     # 执行监控的服务。注意这个设置不能紧挨着写在vrrp_script配置块的后面（实验中碰过的坑），否则nginx监控失效！！
        chk_nginx     # 引用VRRP脚本，即在 vrrp_script 部分指定的名字。定期运行它们来改变优先级，并最终引发主备切换。
    }

    virtual_ipaddress { #VRRP HA 虚拟地址
        192.168.39.111
    }
}
virtual_server 192.168.39.111 80 {
    delay_loop 6
    lb_algo rr
    lb_kind NAT
    nat_mask 255.255.255.0
    persistence_timeout 50
    protocol TCP

    real_server 192.168.39.126 80 {
        weight 1
        HTTP_GET { #连接不稳定问题
            url {
              path /
              
            }
            connect_timeout 3
            nb_get_retry 3
            delay_before_retry 3
        }
    }
}

```

 

两台服务器不同的地方

state MASTER/BACKUP # 指定keepalived的角色，MASTER表示此主机是主服务器，BACKUP表示此主机是备用服务器。     

priority 100 # 定义优先级，数字越大，优先级越高，在同一个vrrp_instance下，MASTER的优先级必须大于BACKUP的优先级，值范围 0-254；     

real_server 192.168.39.126 80 本机地址  

 

nginx 监测脚本 vim /etc/keepalived/nginx_check.sh

  #!/bin/bash  set -x     nginx_status=`ps -C nginx --no-header |wc -l`  if [ ${nginx_status} -eq 0 ];then    service  nginx start    sleep 1       if [  `ps -C nginx --no-header |wc -l` -eq 0 ];then  #nginx重启失败        echo -e "$(date): nginx is  not healthy, try to killall keepalived!"   >> /etc/keepalived/keepalived.log        killall keepalived    fi  fi  echo $?  

 

keepalived服务监测脚本  vim /opt/scripts/keepalived_monitor.sh

```
#!/bin/bash
set -x

nginx_status=`ps -C nginx --no-header |wc -l`
if [ ${nginx_status} -eq 0 ];then
    service nginx start
    sleep 1

    if [ `ps -C nginx --no-header |wc -l` -eq 0 ];then    #nginx重启失败
        echo -e "$(date):  nginx is not healthy, try to killall keepalived!"  >> /etc/keepalived/keepalived.log
        killall keepalived
    fi
fi
echo $?

```

 

### 启动nginx和keepalived

```
service nginx start  # 启动服务

service nginx stop  # 停止服务

service nginx restart # 重启服务
```

```
service keepalived start  # 启动服务

service keepalived stop  # 停止服务

service keepalived restart # 重启服务
```

 

通过VIP来访问nginx，停掉机器上面的keepalived，可以观察到显示结果。
