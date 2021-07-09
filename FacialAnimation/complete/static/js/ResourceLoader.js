function ResourceLoader(url,camera,unitProcess) {
    this.url;//资源路径
    this.camera;
    this.cameraPre;
    this.unitProcess;

    this.init(url,camera,unitProcess);
}
ResourceLoader.prototype={
    init:function (url,camera,unitProcess) {
        this.url=url;
        this.camera=camera;
        this.unitProcess=unitProcess;

        this.cameraPre={};
        var scope=this;
        scope.loadGeometry();
    },

    updateCameraPre:function(){
        this.cameraPre.position=this.camera.position.clone();
        this.cameraPre.rotation=this.camera.rotation.clone();
    },

    cameraHasChanged:function(){
        return this.camera.position.x !== this.cameraPre.position.x ||
            this.camera.position.y !== this.cameraPre.position.y ||
            this.camera.position.z !== this.cameraPre.position.z ||
            this.camera.rotation.x !== this.cameraPre.rotation.x ||
            this.camera.rotation.y !== this.cameraPre.rotation.y ||
            this.camera.rotation.z !== this.cameraPre.rotation.z;
    },

    loadGeometry:function(){
        const order_list = JSON.parse(localStorage.getItem("order"));
        const time_list = JSON.parse(localStorage.getItem("time"));
        console.log(
            order_list,time_list
        );

        var mixer = { };
        initModel();

        function timeLine(){
            var time_line = [];
            time_line.push(0);
            for(let i = 1; i < order_list.length; i++){
                var time = time_line[i-1] + 500 +time_list[i-1];
                time_line.push(time);
            }
            return time_line;
        }
        var time_line = timeLine();

        var clock = new THREE.Clock();
        var past_time = -1000;//第一次播放时，音频会相对动画延迟。
        var upda = 0;
        var animate = function() {
            requestAnimationFrame( animate );
            var time = clock.getDelta() * 1000;
            past_time += time;



            if(past_time > 0) {
                upda = getUpda();
                //console.log("past_time:" + past_time + "  upda:" + upda);
                mixer[upda].update(time/1000);
            }
        };

        function getUpda(){
            var upda = 5;
            var num = order_list.length;
            let i;
            for(i = 0; i < num; i++){
                if(past_time >= time_line[i])
                    continue;
                else break;
            }
            var list_num = i-1;//当前在读哪句话
            var move_list_now = order_list[list_num];//当前这句话的动作序列
            var move_num = move_list_now.length;//当前动作序列的动作数量
            var move_time = time_list[list_num]/move_num;//每个动作花费的时间
            var p_time = past_time-time_line[list_num];//开始读这句话时经过的时间
            //console.log(list_num, move_list_now, move_num, move_time, p_time);
            if(p_time < time_list[list_num]){//当还在读句子时
                upda = move_list_now[parseInt( p_time / move_time )];
            }
            return upda;
        }

        $(function() {
            $('#play').click(function(){
                console.log("按下了play按钮");

                if(past_time < 0){
                    past_time = -1000;
                } else past_time = 0;
                //console.log(order_list.length)
                animate();//开始播放动画

                $.ajax({//这部分代码似乎没有什么用处
                    url: '/play',
                    data:{
                        num: order_list.length,
                    },
                    dataType: 'JSON',
                    type: 'GET',
                    success: function(data){
                        console.log("success");
                    }
                });
                /**/

            });
        });

        function initModel(){//ResourceLoader.js中加载人物模型（function initModel）
            var model_url = "http://127.0.0.1:5000/static/model.gltf";
            //console.log(model_url);
            var loader=new THREE.GLTFLoader();
            loader.load(model_url, function (obj){
                var model = obj.scene;
                model.position.x = 0.05;
                model.position.y = 0.26;
                model.position.z = -268.6;
                model.scale.set(1, 1, 1);
                this.scene.add(model);
                //将模型绑定到动画混合器里面
                var anima = obj.animations;
                mixer[0] = new THREE.AnimationMixer( model );
                mixer[1] = new THREE.AnimationMixer( model );
                mixer[2] = new THREE.AnimationMixer( model );
                mixer[3] = new THREE.AnimationMixer( model );
                mixer[4] = new THREE.AnimationMixer( model );
                mixer[5] = new THREE.AnimationMixer( model );
                //同时将这个外部模型的动画全部绑定到动画混合器里面
                mixer[0].clipAction(anima[0]).play();//b
                mixer[1].clipAction(anima[3]).play();//d
                mixer[2].clipAction(anima[1]).play();//a
                mixer[3].clipAction(anima[6]).play();//e
                mixer[4].clipAction(anima[7]).play();//u
                mixer[5].clipAction(anima[10]).play();//static
                static_animate();
            })
        }

        var static_animate = function (){//播放口型动画（function animate）
            requestAnimationFrame( static_animate );
            mixer[5].update(100)
        }
    },
}
