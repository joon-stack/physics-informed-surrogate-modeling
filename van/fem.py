from anastruct import SystemElements
ss = SystemElements()

E = 2e10
I = 0.0001/12
ss.add_element(location=[[0, 0], [0.2, 0]], EI=E*I)
ss.add_element(location=[[0.2, 0], [0.4, 0]], EI=2*E*I)
ss.add_element(location=[[0.4, 0], [0.6, 0]], EI=3*E*I)
ss.add_element(location=[[0.6, 0], [0.8, 0]], EI=4*E*I)
ss.add_element(location=[[0.8, 0], [1.0, 0]], EI=5*E*I)

ss.add_support_fixed(node_id=1)

# ss.q_load(element_id=[1,2,3,4,5], qw=-1)
ss.point_load(node_id=6, Fy=-50000)
ss.solve()

ss.show_structure()
ss.show_displacement()
ss.show_bending_moment()
x, y= ss.show_displacement(values_only=True)
x, y= ss.show_bending_moment(values_only=True)
print(y)
# print(t)