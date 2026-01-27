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

############################ PRELIMINARY DEFINITIONS
include("./Environnement/Custom_Env_with_initialTMS.jl")

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
#########################


function smoothCHSH(rawCHSH::Vector,partition)
    smCHSHx=Vector{Float64}(undef,0)
    smCHSHy=Vector{Float64}(undef,0)
    l=length(rawCHSH)
    numpartitions=floor(Int64,l/partition)
    
    for i in 1:numpartitions
        smCHSHx=push!(smCHSHx, (partition*i + partition*(i-1))/2)  #we assign the x value to the average between the extremes of that partition
        smCHSHy=push!(smCHSHy, sum(rawCHSH[partition*(i-1)+1:partition*i])/partition)  #every entry of smCHSH is given by the average of a number of "partition" entries of rawCHSH
    end
    
    if mod(l,partition) >0
        smCHSHx=push!(smCHSHx,(l-partition*numpartitions)/2)
        smCHSHy=push!(smCHSHy, sum(rawCHSH[partition*numpartitions+1:end])/mod(l,partition))  #the last element of shCHSH is averaged over the remaining elements of rawCHSH, if any
    end

    return smCHSHx, smCHSHy 
end


#here we load good examples of a learning agent based on different strategies (see paper)

loadStrategy1=load("data/1copypergate-1e-2CHSHoptimthreshold-signbin-4modes4gates-PPOPolicy-anygate-SPprojection-FinalRewardFunc-UpFreq64-TrajectCap256.jld2", "hook")

loadStrategy3=load("data/1copypergate-1e-2CHSHoptimthreshold-signbin-6modes12gates-PPOPolicy-initialize3TMS-SPprojection-FinalRewardFunc-UpFreq64-TrajectCap256-variation1-reducedClipRange.jld2", "hook")

loadStrategy5=load("data/1copypergate-1e-2CHSHoptimthreshold-signbin-6modes27gates-PPOPolicy-initialize6SMS-SPprojection-FinalRewardFunc-UpFreq64-TrajectCap256-variation1-reducedClipRange.jld2", "hook")



rawStrategy1=loadStrategy1.v
rawStrategy3=loadStrategy3.v
rawStrategy5=loadStrategy5.v

smoothStrategy1=smoothCHSH(rawStrategy1,100)
smoothStrategy3=smoothCHSH(rawStrategy3,10)
smoothStrategy5=smoothCHSH(rawStrategy5,1)

#=
strat1plot=plot(smoothStrategy1,
xlabel = "Nr. episodes",
ylabel = "CHSH score",
legend = false)
hline!([2.068],color=:red, linestyle=:dash)

savefig(strat1plot, "../../julia_plots/strategy1plot.pdf")
=#

strat3plot=plot(smoothStrategy3,
xlabel = "Nr. episodes",
ylabel = "CHSH score",
legend = false)
hline!([2.072],color=:red, linestyle=:dash)

savefig(strat3plot, "../../julia_plots/strategy3plot.pdf")
