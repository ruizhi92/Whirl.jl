# IFHERK

"""
    IFHERK(rk::u,f,Δt,plan_intfact,B₁ᵀ,B₂,r₁,r₂;[tol=1e-3])

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
struct IFHERK{FH1,FH2,FR1,FR2,FC,FS,TU,TF}

  # time step size
  Δt :: Float64

  rk :: RKParams
  rkdt :: RKParams

  # Integrating factors
  H :: Vector{FH1}
  H⁻¹ :: Vector{FH2}

  # right hand side in the differential equations
  r₁ :: FR1  # function of u and t, returns TU
  r₂ :: FR2  # function of u and t, returns TF

  # function of u and t, returns B₁ᵀ and B₂
  plan_constraints :: FC

  # Vector of saddle-point systems
  S :: Vector{FS}  # -B₂HB₁ᵀ

  # scratch space
  qᵢ :: TU
  ubuffer :: TU
  fbuffer :: TF
  v̇ :: Vector{TU}

  # iterative solution tolerance
  tol :: Float64

end

function (::Type{IFHERK})(u::TU,f::TF,Δt::Float64,
                          plan_intfact::FI,
                          plan_constraints::FC,
                          rhs::Tuple{FR1,FR2};
                          tol::Float64=1e-3,
                          conditioner::FP = x -> x,
                          rk::RKParams=RK31) where {TU,TF,FI,FC,FR1,FR2,FP}

   # templates for the operators
   # r₁ acts on TU and time, r₂ acts on TU and time
   optypes = ((TU,Float64),(TU,Float64))
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
    fbuffer = deepcopy(f)
    v̇ = [Nodes(u) for i = 1:rk.st]

    # construct an array of operators for the integrating factor. Each
    # one can act on data of type `u` and return data of the same type.
    # e.g. we can call Hlist[1]*u to get the result.
    #---------------------------------------------------------------------------
    dclist = rk.c[2:end]

    if TU <: Tuple
      (FI <: Tuple && length(plan_intfact) == length(u)) ||
                error("plan_intfact argument must be a tuple")
      Hlist = [map((plan,ui) -> plan(dc*Δt,ui),plan_intfact,u) for dc in unique(dclist)]
      H⁻¹list = [map((plan,ui) -> plan(-dc*Δt,ui),plan_intfact,u) for dc in unique(dclist)]

    else
      Hlist = [plan_intfact(dc*Δt,u) for dc in unique(dclist)]
      H⁻¹list = [plan_intfact(-dc*Δt,u) for dc in unique(dclist)]
    end

    H = [Hlist[i] for i in indexin(dclist,unique(dclist))]
    H⁻¹ = [H⁻¹list[i] for i in indexin(dclist,unique(dclist))]

    # preform the saddle-point systems
    # these are overwritten if B₁ᵀ and B₂ vary with time
    Slist = [construct_saddlesys(plan_constraints,H[i],H⁻¹[i],u,f,0.0,
                    tol)[1] for i=1:rk.st]
    S = [Slist[i] for i in indexin(dclist,unique(dclist))]

    h1type,_ = typeof(H).parameters
    h2type,_ = typeof(H⁻¹).parameters
    stype,_ = typeof(S).parameters


    # fuse the time step size into the coefficients for some cost savings
    rkdt = deepcopy(rk)
    rkdt.a .*= Δt
    rkdt.c .*= Δt


    ifherksys = IFHERK{h1type,h2type,typeof(r₁),typeof(r₂),FC,stype,TU,TF}(Δt,rk,rkdt,
                                H,H⁻¹,r₁,r₂,
                                plan_constraints,S,
                                qᵢ,ubuffer,fbuffer,v̇,
                                tol)

    # pre-compile
    #ifherksys(0.0,u)

    return ifherksys
end

function Base.show(io::IO, scheme::IFHERK{FH1,FH2,FR1,FR2,FC,FS,TU,TF}) where {FH1,FH2,FR1,FR2,FC,FS,TU,TF}
    println(io, "Order-$(scheme.rk.st)+ IF-HERK integrator with")
    println(io, "   State of type $TU")
    println(io, "   Force of type $TF")
    println(io, "   Time step size $(scheme.Δt)")
end

# this function will call the plan_constraints function and return the
# saddle point system for a single instance of H, (and B₁ᵀ and B₂)
# plan_constraints should only compute B₁ᵀ and B₂ (and P if needed)
function construct_saddlesys(plan_constraints::FC,H::FH1,H⁻¹::FH2,
                           u::TU,f::TF,t::Float64,tol::Float64) where {FC,FH1,FH2,TU,TF}

    sys = plan_constraints(u,t) # sys contains B₁ᵀ and B₂ before fixing them up

    # B₁ᵀ acts on type TF, B₂ acts on TU
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

    # Actually call SaddleSystem
    if TU <: Tuple
        S = map((ui,fi,Hi,H⁻¹i,B₁ᵀi,B₂i) ->
                  SaddleSystem((ui,fi),(x->H*(B₁ᵀ(x)),x->B₂(H⁻¹*x)),tol=tol,issymmetric=false,isposdef=true,store=true,precompile=false),
                    u,f,H,H⁻¹,B₁ᵀ,B₂)
    else
        S = SaddleSystem((u,f),(x->H*(B₁ᵀ(x)),x->B₂(H⁻¹*x)),tol=tol,
                issymmetric=false,isposdef=true,store=true,precompile=false)
    end

    return S, ops

end


#-------------------------------------------------------------------------------
# Advance the IFHERK solution by one time step
# This form works when u is NOT a tuple
function (scheme::IFHERK{FH1,FH2,FR1,FR2,FC,FS,TU,TF})(t::Float64,u::TU) where
                      {FH1,FH2,FR1,FR2,FC,FS,TU,TF}
  @get scheme (rk,rkdt,H,H⁻¹,plan_constraints,r₁,r₂,qᵢ,ubuffer,fbuffer,v̇,tol)

    # H[i] corresponds to H(cᵢ₊₁Δt)
    # rkdt coefficients includes the time step size
    f = deepcopy(fbuffer)

    # first stage, i = 1
    # initial value of v̇ is set to 0 when creating ifherk object
    i = 1
    tᵢ = t
    qᵢ .= u     # qᵢ is used to store initial value of u

    for i = 2:rk.st+1
        # set time
        tᵢ = t + rkdt.c[i]

        # construct saddlesys
        S, (_, B₂) = construct_saddlesys(plan_constraints,H[i-1],H⁻¹[i-1],u,f,tᵢ,tol)

        # construct lower right hand of saddle system
        fbuffer = B₂(qᵢ)
        fbuffer .-= r₂(u,tᵢ)
        ubuffer.data .= 0.0
        for j = 1:i-2
            ubuffer .+= rkdt.a[i,j] .* v̇[j]
        end
        fbuffer += B₂(H⁻¹[i-1]*ubuffer)
        fbuffer ./= -rkdt.a[i,i-1]

        # construct upper right hand side of saddle system
        ubuffer = H[i-1]*r₁(u,tᵢ)

        # solve the linear system
        v̇[i-1], f = S\(ubuffer,fbuffer)

        # accumulate velocity up to the current stage
        u .= H[i-1]*qᵢ
        for j = 1:i-1
            u .+=  rkdt.a[i,j] .* v̇[j]
        end        
    end

    return tᵢ, u, f
end



#-------------------------------------------------------------------------------
# # Advance the IFHERK solution by one time step
# # This form works when u is a tuple of state vectors
# function (scheme::IFHERK{FH,FR1,FR2,FC,FS,TU,TF})(t::Float64,u::TU) where
#                           {FH,FR1,FR2,FC,FS,TU<:Tuple,TF<:Tuple}
#   @get scheme (rk,rkdt,H,plan_constraints,r₁,r₂,qᵢ,w,fbuffer,ubuffer,tol,
#                 _isstaticconstraints,_issymmetric,_isstored)
#   @get rk (st, c, a)
#
#   # H[i] corresponds to H(i,i+1) = H((cᵢ - cᵢ₋₁)Δt)
#   # Each of the coefficients includes the time step size
#
#   f = deepcopy(fbuffer)
#
#   i = 1
#   tᵢ₊₁ = t
#   for I in eachindex(u)
#     ubuffer[I] .= u[I]
#     qᵢ[I] .= u[I]
#   end
#
#
#   S,_ = construct_saddlesys(plan_constraints,H[i],u,f,tᵢ₊₁,tol,_issymmetric,_isstored)
#
#
#   if st > 1
#     # first stage, i = 1
#
#     w[i] = r₁(u,tᵢ₊₁)
#     ftmp = r₂(u,tᵢ₊₁) # r₂ # seems like a memory re-allocation...
#     for I in eachindex(u)
#       w[i][I] .*= rkdt.a[i,i]  # gᵢ
#
#       ubuffer[I] .+= w[i][I] # r₁ = qᵢ + gᵢ
#       fbuffer[I] .= ftmp[I]
#       # could solve this system for the whole tuple, too...
#       fill!(f[I],0.0)
#       A_ldiv_B!((u[I],f[I]),S[I],(ubuffer[I],fbuffer[I]))
#       ubuffer[I] .= S[I].A⁻¹B₁ᵀf
#       #tmp = S[i][I]\(ubuffer[I],fbuffer[I])  # solve saddle point system
#       #u[I] .= tmp[1]
#       #f[I] .= tmp[2]
#
#       # diffuse the scratch vectors
#       qᵢ[I] .= H[i][I]*qᵢ[I] # qᵢ₊₁ = H(i,i+1)qᵢ
#       w[i][I] .= H[i][I]*w[i][I] # H(i,i+1)gᵢ
#     end
#     tᵢ₊₁ = t + rkdt.c[i]
#
#
#
#     # stages 2 through st-1
#     for i = 2:st-1
#
#       if !_isstaticconstraints
#         S,_ = construct_saddlesys(plan_constraints,H[i],u,f,tᵢ₊₁,tol,_issymmetric,_isstored)
#       else
#         S = scheme.S[i]
#       end
#
#       w[i] = r₁(u,tᵢ₊₁)
#       ftmp = r₂(u,tᵢ₊₁) # r₂
#       for I in eachindex(u)
#         #w[i-1][I] .= (w[i-1][I]-S[i-1][I].A⁻¹B₁ᵀf)./(rkdt.a[i-1,i-1]) # w(i,i-1)
#         w[i-1][I] .= (w[i-1][I]-ubuffer[I])./(rkdt.a[i-1,i-1]) # w(i,i-1)
#
#         w[i][I] .*= rkdt.a[i,i] # gᵢ
#
#         ubuffer[I] .= qᵢ[I] .+ w[i][I] # r₁
#         for j = 1:i-1
#           ubuffer[I] .+= rkdt.a[i,j]*w[j][I] # r₁
#         end
#         fbuffer[I] .= ftmp[I]
#         fill!(f[I],0.0)
#         A_ldiv_B!((u[I],f[I]),S[I],(ubuffer[I],fbuffer[I]))
#         ubuffer[I] .= S[I].A⁻¹B₁ᵀf
#
#         #tmp = S[i][I]\(ubuffer[I],fbuffer[I])  # solve saddle point system
#         #u[I] .= tmp[1]
#         #f[I] .= tmp[2]
#
#         # diffuse the scratch vectors
#         qᵢ[I] .= H[i][I]*qᵢ[I] # qᵢ₊₁ = H(i,i+1)qᵢ
#         for j = 1:i
#           w[j][I] .= H[i][I]*w[j][I] # for j = i, this sets H(i,i+1)gᵢ
#         end
#       end
#       tᵢ₊₁ = t + rkdt.c[i]
#
#     end
#     i = st
#     if !_isstaticconstraints
#       S,_ = construct_saddlesys(plan_constraints,H[i],u,f,tᵢ₊₁,tol,_issymmetric,_isstored)
#     else
#       S = scheme.S[i]
#     end
#     for I in eachindex(u)
#       w[i-1][I] .= (w[i-1][I]-ubuffer[I])./(rkdt.a[i-1,i-1]) # w(i,i-1)
#
#     end
#   end
#
#   # final stage (assembly)
#   w[i] = r₁(u,tᵢ₊₁)
#   ftmp = r₂(u,tᵢ₊₁) # r₂
#   for I in eachindex(u)
#     w[i][I] .*= rkdt.a[i,i]
#
#     ubuffer[I] .= qᵢ[I] .+ w[i][I] # r₁
#     for j = 1:i-1
#       ubuffer[I] .+= rkdt.a[i,j]*w[j][I] # r₁
#     end
#     fbuffer[I] .= ftmp[I]
#     fill!(f[I],0.0)
#     A_ldiv_B!((u[I],f[I]),S[I],(ubuffer[I],fbuffer[I]))
#     f[I] ./= rkdt.a[i,i]
#   end
#   t = t + rkdt.c[i]
#
#
#   return t, u, f
#
# end
