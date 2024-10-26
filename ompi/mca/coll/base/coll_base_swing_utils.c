#include "coll_base_swing_utils.h"

// static inline int get_static_bitmap(unsigned char** send_bitmap, unsigned char** recv_bitmap, int rank, int comm_sz) {
//   switch (comm_sz) {
//     case 2: {
//       *send_bitmap = send_2[rank];
//       *recv_bitmap = recv_2[rank];
//       break;
//     }
//     case 4: {
//       *send_bitmap = send_4[rank];
//       *recv_bitmap = recv_4[rank];
//       break;
//     }
//     case 8: {
//       *send_bitmap = send_8[rank];
//       *recv_bitmap = recv_8[rank];
//       break;
//     }
//     case 16: {
//       *send_bitmap = send_16[rank];
//       *recv_bitmap = recv_16[rank];
//       break;
//     }
//     case 32: {
//       *send_bitmap = send_32[rank];
//       *recv_bitmap = recv_32[rank];
//       break;
//     }
//     default: return -1;
//   }
//   return 0;
// }
