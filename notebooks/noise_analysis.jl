### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 92880eed-92da-44ab-b1b0-b2519aa3c747
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(@__DIR__)
end

# ╔═╡ 15690810-ebd1-11f0-8699-2b34af8b9415
begin
	using QuantumOpticalCircuits
	using SparseArrays
	using LinearAlgebra
	using DomainSets
	using DomainIntegrals
	using Optim
	using DataFrames
	using Plots
	using LaTeXStrings
end

# ╔═╡ f42d0215-3588-45e4-9fae-2593d75c8b3e
md"""
Notebook containing the noise analysis of circuit depicted in Fig.1.

This notebook works with the current LTS version of Julia (1.10 as of 2025). In the `notebooks` directory run
`julia --project=. -e "import Pluto; Pluto.run(notebook="noise_analysis.jl")`
"""

# ╔═╡ c4abcaed-d324-4649-90aa-4cce12cac0ee
md"""
We start with defining a few of the parameters. Found using a previous optimisation, Fig. 2 optimal parameters' are :
- $r_{12}=0.0096$ for the TMS between mode 1,2
- $r_{34}=0.44993$ for the TMS between mode 3,4
- $\theta_{13} = 1.50272$ for the beam splitter mixing mode 1 and 3
- $\theta_{24} = 1.63856$ for the beam splitter mixing mode 2 and 4

We also introduce the function `distance(L)` which converts a distance `L` in km into an efficiency `η`, assuming a loss of $0.2$dB/km.
"""

# ╔═╡ 036289ae-3f68-4c06-a8cf-e026e5a684ca
begin
	# r : squeezing constant
	# T : transmitivity
	r1 = 0.00096; λ1 = tanh(r1);
	r2 = 0.44993; λ2 = tanh(r2); 
	θ1 = 1.50272; T1 = cos(θ1)^2;
	θ2 = 1.63856; T2 = cos(θ2)^2;
	
	# homodyne measurement parameters
	θ_A = [0,π/2]
	θ_B = [-π/4,π/4]

	# distance function, assuming a loss of 0.2dB/km
	distance(L) = 10^(-0.02*L) # in km
end

# ╔═╡ 43664f39-394f-4420-9e0b-a3ee12dbbe5b
md"""
Here we define the function producing the state using the circuit depicted in Fig.1. The parameter $\eta$ correspond to the efficiency of the system. When the efficiency is less that $1$, i.e. when accounting for some losses between Alice and Bob, we use a beam-splitter with an angle $\theta_\eta = \arccos(\sqrt(\eta))$.
"""

# ╔═╡ e038a8d0-4570-49ed-8e12-baed948e9105
function state_prep(r1,r2,θ1,θ2; η=1.0,η_A=1.0)
	if η == 1.0
		s = PseudoGaussianState(4) |> TMS(r1)(1,2) |> TMS(r2)(3,4) |> BS(θ1)(1,3) |> BS(θ2)(2,4) |> Heralding(4,tol=1e-8) |> Heralding(3,tol=1e-8)
	elseif η_A == 1.0
		s = PseudoGaussianState(5) |> TMS(r1)(1,2) |> TMS(r2)(3,4) |> BS(θ1)(1,3) |> BS(θ2)(2,4) |> Heralding(4,tol=1e-8) |> Heralding(3,tol=1e-8) |> BS(acos(√(η)))(1,3)
		for st in s.states
			ptrace!(st,3)
		end
	else
		s = PseudoGaussianState(6) |> TMS(r1)(1,2) |> TMS(r2)(3,4) |> BS(θ1)(1,3) |> BS(θ2)(2,4) |> Heralding(4,tol=1e-8) |> Heralding(3,tol=1e-8) |> BS(acos(√(η)))(1,3) |> BS(acos(√(η_A)))(2,4)
		for st in s.states
			ptrace!(st,4)
			ptrace!(st,3)
		end
	end
	return s
end

# ╔═╡ 13dc9e07-70e8-4690-b3a5-0b2f6a8d9d5a
md"""
We here introduce functions to compute the CHSH score from correlators on the homodyne measurements:    
	- `correlator` computes the correlations of outcomes between mode 1 and 2 of a state, assuming binned outcomes : [0,+∞] → +1, [-∞,0] -> -1.  
	- `chsh` computes the linear sum of the four correlators $A_xB_y$, rotating the measurements basis according the inputs $x,y$.  
"""

# ╔═╡ 8dd5abbe-a672-4eb5-b6dd-e0a9d021ab26
begin
	""" Correlator for state `s` using binned outcomes. Uses an analytical formula. """
	function correlator(s::GaussianState)
	    S_hom = sparse([1,2,3,4],[1,3,2,4],1.0)
	    Γ = S_hom*s.σ*S_hom'
	    Γ = Γ[1:2,1:2]-Γ[1:2,3:4]*inv(Γ[3:4,3:4])*Γ[3:4,1:2]
	    a = Γ[1,1]/4; b = Γ[2,2]/4; c = Γ[1,2]/4;
	    G = 1/(2*√(a*b-c^2))
	    G *= (π/2 - atan(c/√(a*b-c^2)))
	    G *= 1/√(det(4π*inv(Γ)))
	    G *= 4
	    G -= 1
	    return G
	end
	
	function correlator(s::PseudoGaussianState)
	    c = 0.0
	    for i in 1:length(s.prob)
	        c += s.prob[i]*correlator(s.states[i])
	    end
	    c *= s.norm
	    return c
	end
	
	function chsh(s::PseudoGaussianState;A=θ_A,B=θ_B)
		S = 0.
		for x in 1:2
			for y in 1:2
				st = copy(s)
				st = st |> PS(A[x])(1,) |> PS(B[y])(2,)
				#@info st.states[1].σ
				c = correlator(st)
				S += (-1)^((x-1)*(y-1))*c
			end
		end
		return S
	end
	
	function safe_params(x::Vector{Float64})
		# r = 1.15 ensures squeezing below 13dB
		x[1] = x[1] % 1.15
		x[2] = x[2] % 1.15
		return x
	end
	
	function chsh_safe(x::Vector{Float64};η=1.0,η_A=1.0)
		x = safe_params(x)
		try
			s = chsh(state_prep(x[1],x[2],x[3],x[4],η=η,η_A=η_A))
			return s
		catch
			return 0
		end
	end
	
	
end

# ╔═╡ 016acaf1-d301-4d11-8f68-997afa6e8e9b
begin
	""" Optimizing the chsh score from starting parameter x0"""
	function optimised_chsh(;x0=[r1,r2,θ1,θ2],η=1.0,η_A=1.0,iter=500)
		#r = optimize(x -> -chsh_safe(x,η=η,η_A=η_A), x0, )
		r = optimize(x -> -chsh_safe(x,η=η,η_A=η_A), x0, NelderMead(), Optim.Options(iterations=iter))
		return -r.minimum, r.minimizer
	end

	""" Optimize the chsh score with respect to an increasing distance between Alice and Bob. Stops when the score is ≤2. """
	function chsh_with_distance(;x0=[r1,r2,θ1,θ2],step=0.05,L_0=0.0,iter=500)
		df = DataFrame(L=Float64[], S=Float64[], r1=Float64[], r2=Float64[], θ1=Float64[], θ2=Float64[])
		s0 = 2√(2)
		L = L_0
		x = copy(x0)
		while true
			s, params = optimised_chsh(x0=x,η=distance(L),iter=iter)
			@info "$L km : $s"
			s < 2.00+1e-8 && return df
			s > s0 && return df
			push!(df, [L,s,params...])
			L = round(L+step, digits=2)
			x0 = params
			s0 = s
		end
	end

	function chsh_with_distance_AB(;x0=[r1,r2,θ1,θ2],step=0.02)
		df = DataFrame(L=Float64[], S=Float64[], r1=Float64[], r2=Float64[], θ1=Float64[], θ2=Float64[])
		s0 = 2√(2)
		L = 0.0
		while true
			s, params = optimised_chsh(x0=x0,η=distance(L),η_A=distance(L))
			@info "$L km : $s"
			s < 2.00+1e-8 && return df
			s > s0 && return df
			push!(df, [L,s,params...])
			L = round(L+step, digits=2)
			x0 = params
			s0 = s
		end
	end

	""" Post selection probability with respect to the heralding detectors efficiency η. """
	function postselection_with_η(r1,r2,θ1,θ2)
		df = DataFrame(η=Float64[], p=Float64[]) 
		η = 1.0
		state(r1,r2,θ1,θ2) = PseudoGaussianState(4) |> TMS(r1)(1,2) |> TMS(r2)(3,4) |> BS(θ1)(1,3) |> BS(θ2)(2,4) 
		while true
			st = state(r1,r2,θ1,θ2)
			p1 = st |> PhotonDetector(4,η=η)
			p = p1*(st |> Heralding(4) |> PhotonDetector(3,η=η))
			@info "$η : $p"
			push!(df, [η, p])	
			p ≤ 1e-12 && return df
			η = round(η-0.001, digits=4)
			η ≤ 0.01 && return df
		end
	end

	""" CHSH score with respect to the heralding detector efficiency """
	function chsh_with_η(r1,r2,θ1,θ2)
		df = DataFrame(η=Float64[], chsh=Float64[]) 
		η = 1.0
		while true
			s = PseudoGaussianState(4) |> TMS(r1)(1,2) |> TMS(r2)(3,4) |> BS(θ1)(1,3) |> BS(θ2)(2,4) |> Heralding(4,tol=1e-8,η=η) |> Heralding(3,tol=1e-8,η=η)
			c = chsh(s)
			@info "$η : $c"
			push!(df, [η, c])	
			c ≤ 2.0+1e-8 && return df
			η = round(η-0.001, digits=4)
			η ≤ 0.001 && return df
		end
	end
end

# ╔═╡ e45f4aa3-d4a1-4d5b-a5a3-491abf82a987
# ╠═╡ show_logs = false
begin
	df_distance_rough = chsh_with_distance(;step=0.5)
	l_df = last(df_distance_rough)
	df_distance_refined = chsh_with_distance(;x0=[l_df.r1,l_df.r2,l_df.θ1,l_df.θ2],step=0.1,L_0=l_df.L+0.1)
	df_distance = vcat(df_distance_rough, df_distance_refined)
end

# ╔═╡ 62c68253-ddb5-43a7-b920-dd52e83c021e
begin
	plot(df_distance.L,df_distance.S,size=(800,600), xlabel=L"\textrm{Distance~(km)}", ylabel=L"\mathrm{CHSH~Score}", grid=true, legend=false, guidefont=25, xtickfontsize=18,ytickfontsize=18,linewidth=3,xticks=0:8,yticks=2.0:0.01:2.07)
end

# ╔═╡ b0e217cf-1192-48e4-bc48-c42c6fa82476
savefig("Distance.pdf");

# ╔═╡ 07eddf5f-6f68-4dab-be9b-d8dd2cc19c16
# ╠═╡ show_logs = false
df_distance_AB = chsh_with_distance_AB(;step=0.1)

# ╔═╡ 6a1c7985-f14e-4a90-b43c-265e447ec844
begin
	plot(df_distance.L,df_distance.S; linewidth=3,label=L"\textrm{Direct ~Transmission}")
	plot!(2*df_distance_AB.L,df_distance_AB.S, label=L"\textrm{Central~Station}", size=(800,600), xlabel=L"\textrm{Total~Distance~(km)}", ylabel=L"\mathrm{CHSH~Score}", grid=true, legend=true, legendfontsize=18, guidefont=25, xtickfontsize=18,ytickfontsize=18,linewidth=3,xticks=0:10,yticks=2.0:0.01:2.07)
end

# ╔═╡ f6cbbfef-aa6b-4b0e-a141-3f4fb49f67c5
savefig("Totaldistance.pdf");

# ╔═╡ ee69b0fa-f37b-4fcb-b628-39634fe66a5d
# ╠═╡ show_logs = false
begin
	df_postselec = postselection_with_η(r1,r2,θ1,θ2)
	plot(df_postselec.η,df_postselec.p,size=(800,600),xlabel=L"\textrm{Threshold~detector~efficiency}~\eta", ylabel=L"\mathrm{Heralding~probability}",  grid=true, legend=false, guidefont=25, xtickfontsize=18,ytickfontsize=18,linewidth=3, xticks=0:0.1:1.0, yticks=[1e-10,1e-9,1e-8,1e-7,1e-6,1e-5], yaxis=:log10, ylim=[1e-10,1.1e-5])
end

# ╔═╡ 135836d9-ec41-436f-b82b-ebef4f4c1b78
# ╠═╡ show_logs = false
begin
	r = optimised_chsh(;x0=[r1,r2,θ1,θ2],η=1.0,η_A=1.0,iter=500)
	df_c = chsh_with_η(r[2]...)
	plot(df_c.η,df_c.chsh, size=(800,600), xlabel=L"\textrm{Efficiency~(η)}", ylabel=L"\mathrm{CHSH~Score}", grid=true, legend=false, legendfontsize=18, guidefont=25, xtickfontsize=18,ytickfontsize=18,linewidth=3,xticks=0:0.1:1,yticks=2.0678:0.0001:2.0682,xlims=[.05,1.01],ylims=[2.0678,2.0682])
end

# ╔═╡ 08cf7497-e19b-4866-973c-e8dd7bb185f3
savefig("EffThreshold.pdf");

# ╔═╡ f7df9519-f67a-4692-9322-19363452fd93
df_distance

# ╔═╡ Cell order:
# ╟─f42d0215-3588-45e4-9fae-2593d75c8b3e
# ╟─92880eed-92da-44ab-b1b0-b2519aa3c747
# ╟─15690810-ebd1-11f0-8699-2b34af8b9415
# ╟─c4abcaed-d324-4649-90aa-4cce12cac0ee
# ╠═036289ae-3f68-4c06-a8cf-e026e5a684ca
# ╟─43664f39-394f-4420-9e0b-a3ee12dbbe5b
# ╠═e038a8d0-4570-49ed-8e12-baed948e9105
# ╟─13dc9e07-70e8-4690-b3a5-0b2f6a8d9d5a
# ╠═8dd5abbe-a672-4eb5-b6dd-e0a9d021ab26
# ╠═016acaf1-d301-4d11-8f68-997afa6e8e9b
# ╠═e45f4aa3-d4a1-4d5b-a5a3-491abf82a987
# ╠═62c68253-ddb5-43a7-b920-dd52e83c021e
# ╠═b0e217cf-1192-48e4-bc48-c42c6fa82476
# ╠═07eddf5f-6f68-4dab-be9b-d8dd2cc19c16
# ╠═6a1c7985-f14e-4a90-b43c-265e447ec844
# ╠═f6cbbfef-aa6b-4b0e-a141-3f4fb49f67c5
# ╠═ee69b0fa-f37b-4fcb-b628-39634fe66a5d
# ╠═135836d9-ec41-436f-b82b-ebef4f4c1b78
# ╠═08cf7497-e19b-4866-973c-e8dd7bb185f3
# ╠═f7df9519-f67a-4692-9322-19363452fd93
