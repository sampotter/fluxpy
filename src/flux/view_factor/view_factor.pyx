cdef extern from "view_factor.h":
    double ff_narayanaswamy_impl(const double tri1[3][3],
                                 const double tri2[3][3])

cpdef ff_narayanaswamy(double[:, ::1] V1, double[:, ::1] V2):
    return ff_narayanaswamy_impl(<const double(*)[3]>&V1[0, 0],
                                 <const double(*)[3]>&V2[0, 0])
