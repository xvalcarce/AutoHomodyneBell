using ReinforcementLearning
using StableRNGs
using Distributions
using Flux
using Flux.Losses
using JLD2
using Logging
using Dates
using FileIO
using Plots
import TensorBoardLogger: TBLogger, @info

include("../Environnement/Custom_Env_with_initialSMS_and_DisplacementBeforeSPprojection.jl")

seed = rand(1:5000)
N = 6  # total number of modes
n_circuit = 20  # max length of the circuit = max number of nodes in the circuit (it should be >=N, otherwise most of the time the heralding probability is zero, i.e. below tolerance)
N_ENV = 1  # the number of environments that will run in parallel (one agent learning but playing in parallel on N_ENV environments)
N_HIDDEN = [150,90] # params per hidden layers
	# hidden 1 : 45 used for 4 modes and all gates / 150 was used for 6 modes and only passive gates
    # hidden 2 : 30 used for 4 modes and all gates / 90 was used for 6 modes and only passive gates	

env = Env(N,n_circuit)

rng = StableRNG(seed)
n_0 = floor(Int64,4*(4+1)/2 + 1) # number of parameters per gaussian state (upper triangle of CM plus weight); needs to be modified if displacement is added!

ns = n_0 * 2^(N-2) #number of parameters for a PseudoGaussian state: the number of states in the mixture is equal to 2^(number of heralded modes on click); this is the number of neurons in the input layer
na = length(env.actions) #total number of actions; this is the number of neurons in the output layer

UPDATE_FREQ = 64 #used to be 256  #update frequency of the neural network, needs to be a power of 2
TRAJECTORY_CAPACITY = 256 #size of memory storage from which the agent updates the policy; should be longer than update frequency to account for good circuits found at the beginning

mutable struct FinalRewardPerEpisode <: AbstractHook
    v::Vector{Float64}   #the CHSH values found by the hook, after optimizing over the circuit parameters after each episode; if =0, then the optimization failed
    circuit::Vector{Vector{node}}  # the circuit config that achieves those CHSH values
    params::Vector{Vector{Float64}} # the parameters of the nodes in the circuit
end

function FinalRewardPerEpisode()
    v=Vector{Float64}(undef,0)
    circuit = Vector{Vector{node}}(undef,0)
    params = Vector{Vector{Float64}}(undef,0)
    return FinalRewardPerEpisode(v,circuit,params)
end

function TensorHook(agent,env;save_dir=nothing)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "Logs", "PPO_CHSH_$(t)")
    end
    lg = TBLogger(save_dir, min_level = Logging.Info)
    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                @info(
                    "training",
                    actor_loss = agent.policy.actor_loss[end][end],
                    critic_loss = agent.policy.critic_loss[end][end], 
                    entropy_loss = agent.policy.entropy_loss[end][end],
                    loss = agent.policy.loss[end][end],
                )
                if is_terminated(env) && length(total_reward_per_episode.rewards) > 0
                    @info "training" reward = total_reward_per_episode.rewards[end] log_step_increment = 0
                end
            end
        end,
    )
end

function (h::FinalRewardPerEpisode)(::PostEpisodeStage, policy, env)  #we always save the circuits in the hook, regardless of the CHSH value
    if (is_terminated(env))
        push!(h.v,env.chsh_absvalue) 
        push!(h.circuit,env.circuit)
        push!(h.params,deepcopy(env.params))
    end
end

function train_agent(N::Int64,agent,env)
    run(agent,env,StopAfterEpisode(N),FinalRewardPerEpisode())
end

function build_nn(input_dim, h_dims, output_dims; rng)
	layers = Vector{Any}()
	# input to first hidden layer
	push!(layers, Dense(input_dim, hidden_dims[1], relu; init = glorot_uniform(rng)))
    # hidden to hidden
    for i in 1:length(hidden_dims)-1
        push!(layers,
              Dense(hidden_dims[i], hidden_dims[i+1], relu; init = glorot_uniform(rng)))
    end
    # last hidden to output
    push!(layers, Dense(hidden_dims[end], output_dim; init = glorot_uniform(rng)))
    return Chain(layers...)
end


function generate_agent(;n_hidden=N_HIDDEN)
	agent = Agent(
           policy = PPOPolicy(
            	approximator = ActorCritic(
                	actor = build_nn(ns, n_hidden, na; rng = rng),
					critic = build_nn(ns, n_hidden, 1;  rng = rng),
                	optimizer = ADAM(1e-4), #until 22/02/2024 it was 1e-5
            	) |> cpu, 
            	γ = 0.99f0, 
            	λ = 0.95f0,
            	clip_range = 0.1f0, #value used until 30/062024: 0.2f0 -- original value: 0.1f0,
            	max_grad_norm = 0.5f0,#0.5f0,
            	n_epochs = 25,
            	n_microbatches = 4,
            	actor_loss_weight = 1.0f0,#1.0f0,
            	critic_loss_weight = 0.5f0,#0.5f0,
            	entropy_loss_weight = 0.001f0,#0.001f0,
            	update_freq = UPDATE_FREQ,
        	),
        	trajectory = PPOTrajectory(;
            	capacity = TRAJECTORY_CAPACITY,
            	state = Matrix{Float32} => (ns, N_ENV),
            	action = Vector{Int} => (N_ENV,),
            	action_log_prob = Vector{Float32} => (N_ENV,),
            	reward = Vector{Float32} => (N_ENV,),
            	terminal = Vector{Bool} => (N_ENV,),
        	),
    	)
	return agent
end

function train(env,N::Int64)
	agent = generate_agent()
	stop_condition = StopAfterEpisode(N)
    hook = FinalRewardPerEpisode()
    ex = Experiment(agent, env, stop_condition, hook, "Homodyne_measurements")
    run(ex)
	return agent, hook
end

function run_trainedagent(N::Int64,agent,env)
    learnedpolicy = agent.policy
    stop_condition = StopAfterEpisode(N)
    hook = FinalRewardPerEpisode()
    run(learnedpolicy, env, stop_condition, hook)
    return hook
end



### training
Nep = 520 # max number of episodes 
agent,hook = train(env,Nep)
save("data/1copypergate-1e-2CHSHoptimthreshold-signbin-6modes20gates-PPOPolicy-initialize6SMS-SPprojection-FinalRewardFunc-UpFreq64-TrajectCap256-variation1-reducedClipRange-displacement.jld2", Dict("agent" => agent, "hook" => hook))

#=
#### running the trained agent
Nep=100
trainedagent=load("data/1copypergate-1e-2CHSHoptimthreshold-signbin-6modes12gates-PPOPolicy-initialize3TMS-SPprojection-FinalRewardFunc-UpFreq64-TrajectCap256-variation1-reducedClipRange.jld2", "agent")
results=run_trainedagent(Nep,trainedagent,env)
save("data/run/TRAINED-1copypergate-1e-2CHSHoptimthreshold-signbin-6modes12gates-PPOPolicy-initialize3TMS-SPprojection-FinalRewardFunc-UpFreq64-TrajectCap256-variation1-reducedClipRange.jld2", Dict("hook" => results))
=#

#### Checking for hooks
output = load("data/1copypergate-1e-2CHSHoptimthreshold-signbin-6modes20gates-PPOPolicy-initialize6SMS-SPprojection-FinalRewardFunc-UpFreq64-TrajectCap256-variation1-reducedClipRange-displacement.jld2", "hook")
plot(output.v)
