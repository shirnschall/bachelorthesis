order=3
VT = HDiv(mesh, order=order, dirichlet="wall|inlet|cyl")
VF = TangentialFacetFESpace(mesh, order=order, dirichlet="wall|inlet|cyl")
Q = L2(mesh, order=order-1)
X = VT*VF*Q

u, uhat, p = X.TrialFunction()
v, vhat, q = X.TestFunction()

n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size
dS = dx(element_boundary=True)

nu = 1e-3

def tang(vec):
    return vec - (vec*n)*n

# Thesis Christoph Lehrenfeld, page 71
stokes = nu*InnerProduct(Grad(u), Grad(v)) * dx \
    + nu*InnerProduct(Grad(u)*n, tang(vhat-v)) * dS \
    + nu*InnerProduct(Grad(v)*n, tang(uhat-u)) * dS \
    + nu*4*order*order/h * InnerProduct(tang(vhat-v), tang(uhat-u)) * dS \
    + div(u)*q*dx + div(v)*p*dx -1e-11/nu*p*q*dx

a = BilinearForm (stokes).Assemble()