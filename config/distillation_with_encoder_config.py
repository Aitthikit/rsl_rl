
@configclass
class GmtRobotDistillationVeloEnvCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 120
    max_iterations = 1000
    save_interval = 50
    experiment_name = "GMT_distillation"
    empirical_normalization = False
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_hidden_dims=[128, 128, 128],
        teacher_hidden_dims=[128, 128, 128],
        activation="elu",
        # rnn_hidden_dim=128,
        # rnn_type="lstm",
        # rnn_num_layers=1,
        # teacher_recurrent=True,
        encoder_obs = True,
    )
    encoder = RslRlEncoderDistillationCfg(
        student_type = "convgru",
        student_gru_hidden_size = 256,
        student_gru_num_layers = 1,
        student_conv_channels = [16, 32],
        student_conv_kernel_sizes = [3, 3],
        student_pool_sizes = [2, 2, 2],
        student_hidden_dims=[256,128],

        teacher_type = "gru",
        teacher_gru_hidden_size = 256,
        teacher_gru_num_layers = 2,
        teacher_hidden_dims=[256,128],

        output_dim=8,
        obs_indices = 36,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
   
    )

@configclass
class GmtRobotDistillationVeloDPPOEnvCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 120
    max_iterations = 1000
    save_interval = 50
    experiment_name = "GMT_distillation_RNN"
    empirical_normalization = False
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_hidden_dims=[128, 128, 128],
        teacher_hidden_dims=[128, 128, 128],
        activation="elu",
        rnn_hidden_dim=128,
        rnn_type="lstm",
        rnn_num_layers=1,
        teacher_recurrent=True,
        encoder_obs = True,
    )
    encoder = RslRlEncoderDistillationCfg(
        student_type = "convgru",
        student_gru_hidden_size = 256,
        student_gru_num_layers = 1,
        student_conv_channels = [16, 32],
        student_conv_kernel_sizes = [3, 3],
        student_pool_sizes = [2, 2, 2],
        student_hidden_dims=[256,128],

        teacher_type = "gru",
        teacher_gru_hidden_size = 256,
        teacher_gru_num_layers = 2,
        teacher_hidden_dims=[256,128],

        output_dim=8,
        obs_indices = 36,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
   
    )
