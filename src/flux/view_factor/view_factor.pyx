cdef extern from "view_factor.h":
    double ff_narayanaswamy_impl(const double tri1[3][3],
                                 const double n1[3],
                                 const double tri2[3][3],
                                 double area1)

cpdef ff_narayanaswamy(double[:, ::1] V1, double[::1] n1,
                           double[:, ::1] V2, double area1):
    return ff_narayanaswamy_impl(
        <const double(*)[3]>&V1[0, 0],
        &n1[0],
        <const double(*)[3]>&V2[0, 0],
        area1);
