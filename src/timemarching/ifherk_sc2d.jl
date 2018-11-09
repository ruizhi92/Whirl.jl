# IFHERK_sc2d

"""
    IFHERK_sc2d(u,f,Δt,plan_intfact,B₁ᵀ,B₂,r₁,r₂;[tol=1e-3],[issymmetric=false],[rk::RKParams=RK31])

Construct an integrator to advance a system of the form

du/dt - Au = -B₁ᵀf + r₁(u,t)
B₂u = r₂(u,t)

The resulting integrator will advance the system `(u,f)` by one time step, `Δt`.
The optional argument `tol` sets the tolerance of iterative saddle-point solution,
if applicable.

# Arguments

- `u` : example of state vector data
- `f` : example of constraint force vector data
- `Δt` : time-step size
- `plan_intfact` : constructor to set up integrating factor operator for `A` that
              will act on type `u` (by left multiplication) and return same type as `u`
- `plan_constraints` : constructor to set up the
- `B₁ᵀ` : operator acting on type `f` and returning type `u`
- `B₂` : operator acting on type `u` and returning type `f`
- `r₁` : operator acting on type `u` and `t` and returning `u`
- `r₂` : operator acting on type `u` and `t` and returning type `f`
"""
struct IFHERK_sc2d{NS,FH,FR1,FR2,FC,FS,TU,TF}

  # time step size
  Δt :: Float64

  rk :: RKParams
  rkdt :: RKParams

  # Integrating factors
  H :: Vector{FH}

  r₁ :: FR1  # function of u and t, returns TU
  r₂ :: FR2  # function of u and t, returns TF

  # function of u and t, returns B₁ᵀ and B₂
  plan_constraints :: FC

  # Vector of saddle-point systems
  S :: Vector{FS}  # -B₂HB₁ᵀ

  # scratch space
  qᵢ :: TU
  ubuffer :: TU
  w :: Vector{TU}
  fbuffer :: TF

  # iterative solution tolerance
  tol :: Float64

  # flags
  _issymmetric :: Bool # is the system matrix symmetric?
  _isstaticconstraints :: Bool  # do the system constraint operators stay unchanged?
  _isstaticmatrix :: Bool # is the upper left-hand matrix static?
  _isstored :: Bool # is the Schur complement matrix (inverse) stored?

end

function (::Type{IFHERK_sc2d})(u::TU,f::TF,Δt::Float64,
                          plan_intfact::FI,
                          plan_constraints::FC,
                          rhs::Tuple{FR1,FR2},
                          coord::Array{Float64,2};
                          tol::Float64=1e-3,
                          conditioner::FP = x -> x,
                          issymmetric::Bool=false,
                          isstaticconstraints::Bool=true,
                          isstaticmatrix::Bool=true,
                          isstored::Bool=false,
                          isinit::Bool=false,
                          rk::RKParams{NS}=RK31) where {TU,TF,FI,FC,FR1,FR2,FP,NS}


   # templates for the operators
   # r₁ acts on TU and time
   # r₂ acts on TU and time
   optypes = ((TU,Float64),(TU,Float64,Array{Float64,2}))
   opnames = ("r₁","r₂")
   ops = []
   # check for methods for r₁ and r₂
   for (i,typ) in enumerate(optypes)
     if method_exists(rhs[i],typ)
       push!(ops,rhs[i])
     else
       error("No valid operator for $(opnames[i]) supplied")
     end
   end
   r₁, r₂ = ops

    # scratch space
    qᵢ = deepcopy(u)
    ubuffer = deepcopy(u)
    w = [deepcopy(u) for i = 1:NS] # one extra for last step in tuple form
    fbuffer = deepcopy(f)

    dclist = diff([0;rk.c])

    # construct an array of operators for the integrating factor. Each
    # one can act on data of type `u` and return data of the same type.
    # e.g. we can call Hlist[1]*u to get the result.
    #---------------------------------------------------------------------------
    if TU <: Tuple
      (FI <: Tuple && length(plan_intfact) == length(u)) ||
                error("plan_intfact argument must be a tuple")
      Hlist = [map((plan,ui) -> plan(dc*Δt,ui),plan_intfact,u) for dc in unique(dclist)]
    else
      Hlist = [plan_intfact(dc*Δt,u) for dc in unique(dclist)]
    end

    H = [Hlist[i] for i in indexin(dclist,unique(dclist))]

    # preform the saddle-point systems
    # these are overwritten if B₁ᵀ and B₂ vary with time
    Slist = [construct_saddlesys(plan_constraints,Hi,u,f,0.0,coord,
                    tol,issymmetric,isstored,precompile=false)[1] for Hi in Hlist]
    S = [Slist[i] for i in indexin(dclist,unique(dclist))]

    htype,_ = typeof(H).parameters
    stype,_ = typeof(S).parameters


    # fuse the time step size into the coefficients for some cost savings
    rkdt = deepcopy(rk)
    rkdt.a .*= Δt
    rkdt.c .*= Δt


    ifherksys = IFHERK_sc2d{NS,htype,typeof(r₁),typeof(r₂),FC,stype,TU,TF}(Δt,rk,rkdt,
                                H,r₁,r₂,
                                plan_constraints,S,
                                qᵢ,ubuffer,w,fbuffer,
                                tol,issymmetric,isstaticconstraints,isstaticmatrix,isstored)

    # pre-compile
    #ifherksys(0.0,u)

    return ifherksys
end

function Base.show(io::IO, scheme::IFHERK_sc2d{NS,FH,FR1,FR2,FC,FS,TU,TF}) where {NS,FH,FR1,FR2,FC,FS,TU,TF}
    println(io, "Order-$NS IF-HERK for fs interaction integrator with")
    println(io, "   State of type $TU")
    println(io, "   Force of type $TF")
    println(io, "   Time step size $(scheme.Δt)")
end

# this function will call the plan_constraints function and return the
# saddle point system for a single instance of H, (and B₁ᵀ and B₂)
# plan_constraints should only compute B₁ᵀ and B₂ (and P if needed)
function construct_saddlesys(plan_constraints::FC,H::FH,
                           u::TU,f::TF,t::Float64,coord::Array{Float64,2},tol::Float64,issymmetric::Bool,isstored::Bool;
                           precompile::Bool=false) where {FC,FH,TU,TF}

    sys = plan_constraints(u,t,coord) # sys contains B₁ᵀ and B₂ before fixing them up

    # B₁ᵀ acts on type TF
    # B₂ acts on TU
    optypes = ((TF,),(TU,))
    opnames = ("B₁ᵀ","B₂")
    ops = []
    # check for methods for B₁ᵀ and B₂
    for (i,typ) in enumerate(optypes)
      if TU <: Tuple
        opsi = ()
        for I in eachindex(sys[i])
          typI = (typ[1].parameters[I],)
          if method_exists(sys[i][I],typI)
            opsi = (opsi...,sys[i][I])
          elseif method_exists(*,(typeof(sys[i][I]),typI...))
            # generate a method that acts on TU
            opsi = (opsi...,x->sys[i][I]*x)
          else
            error("No valid operator for $(opnames[i]) supplied")
          end
        end
        push!(ops,opsi)
      else
        if method_exists(sys[i],typ)
          push!(ops,sys[i])
        elseif method_exists(*,(typeof(sys[i]),typ...))
          # generate a method that acts on TU
          push!(ops,x->sys[i]*x)
        else
          error("No valid operator for $(opnames[i]) supplied")
        end
      end
    end
    B₁ᵀ, B₂ = ops

    if TU <: Tuple
      S = map((ui,fi,Hi,B₁ᵀi,B₂i) ->
                  SaddleSystem((ui,fi),(Hi,B₁ᵀi,B₂i),tol=tol,issymmetric=issymmetric,isposdef=true,store=isstored,precompile=precompile),
                    u,f,H,B₁ᵀ,B₂)
    else
      S = SaddleSystem((u,f),(H,B₁ᵀ,B₂),tol=tol,
                issymmetric=issymmetric,isposdef=true,store=isstored,precompile=precompile)
    end

    return S, ops



end

"""
    (scheme::IFHERK_sc2d)(u,t,bkins)

Advance IFHERK_sc2d object with body points coords and motions of all stages used
in IFHERK_sc2d. bkins is a vector of 2d Array, which contains coords and motions
as 2d matrix.
"""
function (scheme::IFHERK_sc2d{NS,FH,FR1,FR2,FC,FS,TU,TF})(t::Float64,u::TU,
            bkins::Vector{Array{Float64,2}}) where {NS,FH,FR1,FR2,FC,FS,TU,TF}
  @get scheme (rk,rkdt,H,plan_constraints,r₁,r₂,qᵢ,w,fbuffer,ubuffer,tol,
                    _isstaticconstraints,_issymmetric,_isstored)


  # By Ruizhi, create a container for f info output at all stages
  fs = Vector{Array{Float64,2}}(NS)

  # H[i] corresponds to H(i,i+1) = H((cᵢ - cᵢ₋₁)Δt)
  # Each of the coefficients includes the time step size
  f = deepcopy(fbuffer)

  i = 1
  tᵢ₊₁ = t
  ubuffer .= u
  qᵢ .= u

  if !_isstaticconstraints
    # input coord of stage 1
    S,_ = construct_saddlesys(plan_constraints,H[i],u,f,tᵢ₊₁,bkins[i][:,1:2],tol,_issymmetric,_isstored)
  else
    error("_isstaticconstraints should be set to false for this problem")
  end


  if NS > 1
    # first stage, i = 1

    w[i] .= rkdt.a[i,i].*r₁(u,tᵢ₊₁) # gᵢ
    ubuffer .+= w[i] # r₁ = qᵢ + gᵢ
    fbuffer .= r₂(u,tᵢ₊₁,bkins[i][:,3:4]) # r₂, input motion
    u, f = S\(ubuffer,fbuffer)  # solve saddle point system
    fs[1] = [f.u f.v] ./ rkdt.a[1,1]
    ubuffer .= S.A⁻¹B₁ᵀf
    tᵢ₊₁ = t + rkdt.c[i]

    # diffuse the scratch vectors
    qᵢ .= H[i]*qᵢ # qᵢ₊₁ = H(i,i+1)qᵢ
    w[i] .= H[i]*w[i] # H(i,i+1)gᵢ



    # stages 2 through NS-1
    for i = 2:NS-1
      if !_isstaticconstraints
        S,_ = construct_saddlesys(plan_constraints,H[i],u,f,tᵢ₊₁,bkins[i][:,1:2],tol,_issymmetric,_isstored)
      else
        error("_isstaticconstraints should be set to false for this problem")
      end
      w[i-1] .= (w[i-1]-ubuffer)./(rkdt.a[i-1,i-1]) # w(i,i-1)
      #w[i-1] .= (w[i-1]-S[i-1].A⁻¹B₁ᵀf)./(rkdt.a[i-1,i-1]) # w(i,i-1)
      w[i] .= rkdt.a[i,i].*r₁(u,tᵢ₊₁) # gᵢ
      ubuffer .= qᵢ .+ w[i] # r₁
      for j = 1:i-1
        ubuffer .+= rkdt.a[i,j]*w[j] # r₁
      end
      fbuffer .= r₂(u,tᵢ₊₁,bkins[i][:,3:4]) # r₂, input motion
      u, f = S\(ubuffer,fbuffer)  # solve saddle point system
      fs[i] = [f.u f.v] ./ rkdt.a[i,i]
      ubuffer .= S.A⁻¹B₁ᵀf
      tᵢ₊₁ = t + rkdt.c[i]

      #A_ldiv_B!((u,f),S[i],(ubuffer,fbuffer)) # solve saddle point system

      # diffuse the scratch vectors
      qᵢ .= H[i]*qᵢ # qᵢ₊₁ = H(i,i+1)qᵢ
      for j = 1:i
        w[j] .= H[i]*w[j] # for j = i, this sets H(i,i+1)gᵢ
      end
    end

    i = NS
    if !_isstaticconstraints
      S,_ = construct_saddlesys(plan_constraints,H[i],u,f,tᵢ₊₁,bkins[i][:,1:2],tol,_issymmetric,_isstored)
    else
      error("_isstaticconstraints should be set to false for this problem")
    end
    #w[i-1] .= (w[i-1]-S[i-1].A⁻¹B₁ᵀf)./(rkdt.a[i-1,i-1]) # w(i,i-1)
    w[i-1] .= (w[i-1]-ubuffer)./(rkdt.a[i-1,i-1]) # w(i,i-1)
  end

  # final stage (assembly)
  ubuffer .= qᵢ .+ rkdt.a[i,i].*r₁(u,tᵢ₊₁) # r₁
  for j = 1:i-1
    ubuffer .+= rkdt.a[i,j]*w[j] # r₁
  end
  fbuffer .= r₂(u,tᵢ₊₁,bkins[i][:,3:4]) # r₂
  u, f = S\(ubuffer,fbuffer)  # solve saddle point system
  #A_ldiv_B!((u,f),S[i],(ubuffer,fbuffer)) # solve saddle point system
  f ./= rkdt.a[i,i]
  fs[NS] = [f.u f.v]
  t = t + rkdt.c[i]

  return t, u, f, fs

end
