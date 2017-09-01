./tools/caffe.cpp
// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;
./build/tools/caffe train --solver= --snapshot
int main(int argc, char** argv) {
      return GetBrewFunction(caffe::string(argv[1]))();//g_brew_map保存着所有函数名和函数的入口。
      //训练过程
      int train() {
        /*检查并读取输入参数*/
        CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";//FLAGS_solver --solver这个选项
        CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
            << "Give a snapshot to resume training or weights to finetune "
            "but not both.";
        vector<string> stages = get_stages_from_flags(); //caffe::Phase get_phase_from_flags(caffe::Phase default_value)
        caffe::SolverParameter solver_param;// ./src/caffe/proto/caffe.proto  message SolverParameter 
        caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);// ./util/upgrade_proto.cpp
        solver_param.mutable_train_state()->set_level(FLAGS_level);
        for (int i = 0; i < stages.size(); i++) {
          solver_param.mutable_train_state()->add_stage(stages[i]);
        }
        if (FLAGS_gpu.size() == 0
            && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
            if (solver_param.has_device_id()) {
                FLAGS_gpu = "" +
                    boost::lexical_cast<string>(solver_param.device_id());
            } else {  // Set default GPU if unspecified
                FLAGS_gpu = "" + boost::lexical_cast<string>(0);
            }
        }
        vector<int> gpus;
        get_gpus(&gpus);
        if (gpus.size() == 0) {
          LOG(INFO) << "Use CPU.";
          Caffe::set_mode(Caffe::CPU);
        } else {
          ostringstream s;
          for (int i = 0; i < gpus.size(); ++i) {
            s << (i ? ", " : "") << gpus[i];
          }
          LOG(INFO) << "Using GPUs " << s.str();
      #ifndef CPU_ONLY
          cudaDeviceProp device_prop;
          for (int i = 0; i < gpus.size(); ++i) {
            cudaGetDeviceProperties(&device_prop, gpus[i]);
            LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
          }
      #endif
          solver_param.set_device_id(gpus[0]);
          Caffe::SetDevice(gpus[0]);
          Caffe::set_mode(Caffe::GPU);
          Caffe::set_solver_count(gpus.size());
        }
        //建立来自用户的信号接口，各种交互操作
        caffe::SignalHandler signal_handler(
              GetRequestedAction(FLAGS_sigint_effect),
              GetRequestedAction(FLAGS_sighup_effect));//
                      // caffe::SolverAction::Enum GetRequestedAction(
                      //     const std::string& flag_value) {
                      //   if (flag_value == "stop") {
                      //     return caffe::SolverAction::STOP;
                      //   }
                      //   if (flag_value == "snapshot") {
                      //     return caffe::SolverAction::SNAPSHOT;
                      //   }
                      //   if (flag_value == "none") {
                      //     return caffe::SolverAction::NONE;
                      //   }
                      //   LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
                      // }
        //构造solver类
        //载入param，//Solver类的构造函数，solver是一个实例。
        shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
        solver->SetActionFunction(signal_handler.GetActionFunction());        
        //solver读取网络权重
        if (FLAGS_snapshot.size()) {
          LOG(INFO) << "Resuming from " << FLAGS_snapshot;
          solver->Restore(FLAGS_snapshot.c_str()); // the snapshot solver state to resume training.
        } else if (FLAGS_weights.size()) {
          CopyLayers(solver.get(), FLAGS_weights);// 加载预训练的网络权重
        }
        //开始训练
        if (gpus.size() > 1) {
          //GPU 模式下
          caffe::P2PSync<float> sync(solver, NULL, solver->param());//P2PSync类的构造函数，./src/caffe/parallel.cpp
          sync.Run(gpus);//P2PSync 的 对象sync执行Run函数
                void P2PSync<Dtype>::Run(const vector<int>& gpus) {
                    vector<shared_ptr<P2PSync<Dtype> > > syncs(gpus.size());
                    Prepare(gpus, &syncs);//每个GPU一个sync
                    LOG(INFO)<< "Starting Optimization";
                    //启动线程
                    for (int i = 1; i < syncs.size(); ++i) {
                      syncs[i]->StartInternalThread();//./src/caffe/internal_thread.cpp
                    }
                    // Run root solver on current thread
                    solver->Solve();//此处和CPU调用同样的函数。
                    //结束线程
                    for (int i = 1; i < syncs.size(); ++i) {
                      syncs[i]->StopInternalThread();//./src/caffe/internal_thread.cpp
                    }
                  }

        } else {
          //cpu 模式下
          LOG(INFO) << "Starting Optimization";
          solver->Solve();//调用solve函数，不是构造函数。
          //~/caffe/src/caffe/solver.cpp
            Solve
                Step
                  TestAll
                      TestDetection
        }
        LOG(INFO) << "Optimization Done.";
        return 0;
      }
