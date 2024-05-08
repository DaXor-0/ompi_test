#include "ompi_config.h"

#include "mpi.h"
#include "opal/util/bit_ops.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/op/op.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "coll_base_topo.h"
#include "coll_base_util.h"

int ompi_coll_swift_allreduce(const void *send_buffer, void *recieve_buffer, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module) {
  printf("you're running the correct swift algorithm");  

  int return_value, line, rank, size, adjsize, remote, distance;
  int newrank, newremote, extra_ranks;
  char *tmpsend = NULL, *tmprecv = NULL, *tmpswap = NULL, *inplacebuf_free = NULL, *inplacebuf;
  ptrdiff_t span, gap = 0;

  size = ompi_comm_size(comm);
  rank = ompi_comm_rank(comm);

  /* Allocate and initialize temporary send buffer */
  span = opal_datatype_span(&dtype->super, count, &gap);
  inplacebuf_free = (char*) malloc(span);
  inplacebuf = inplacebuf_free - gap;

  if (MPI_IN_PLACE == send_buffer) {
    return_value = ompi_datatype_copy_content_same_ddt(dtype, count, inplacebuf, (char*)recieve_buffer);
    if (return_value < 0) { line = __LINE__; goto error_hndl; }
  } else {
    return_value = ompi_datatype_copy_content_same_ddt(dtype, count, inplacebuf, (char*)send_buffer);
    if (return_value < 0) { line = __LINE__; goto error_hndl; }
  }


  // Initialize temporary buffers
  tmpsend = (char*) inplacebuf;
  tmprecv = (char*) recieve_buffer;

  // Determine nearest power of two less than or equal to size
  adjsize = opal_next_poweroftwo (size);
  adjsize >>= 1;

  // Calculate extra ranks
  extra_ranks = size - adjsize;

  /* Handle non-power-of-two case:
      - Even ranks less than 2 * extra_ranks send their data to (rank + 1), and
      sets new rank to -1.
      - Odd ranks less than 2 * extra_ranks receive data from (rank - 1),
      apply appropriate operation, and set new rank to rank/2
      - Everyone else sets rank to rank - extra_ranks
  */
  if (rank <  (2 * extra_ranks)) {
    if (0 == (rank % 2)) {
      return_value = MCA_PML_CALL(send(tmpsend, count, dtype, (rank + 1), MCA_COLL_BASE_TAG_ALLREDUCE, MCA_PML_BASE_SEND_STANDARD, comm));
      if (MPI_SUCCESS != return_value) { line = __LINE__; goto error_hndl; }
      newrank = -1;
    }
    else
    {
      return_value = MCA_PML_CALL(recv(tmprecv, count, dtype, (rank - 1),MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE));
      if (MPI_SUCCESS != return_value) { line = __LINE__; goto error_hndl; }
      /* tmpsend = tmprecv (op) tmpsend */
      ompi_op_reduce(op, tmprecv, tmpsend, count, dtype);
      newrank = rank >> 1;
    }
  } else {
    newrank = rank - extra_ranks;
  }

  /* Communication/Computation loop
      - Exchange message with remote node.
      - Perform appropriate operation taking in account order of operations:
      result = value (op) result
  */
  for (distance = 0x1; distance < adjsize; distance <<=1) {
    if (newrank < 0) break;
    /* Determine remote node */
    newremote = newrank ^ distance;
    remote = (newremote < extra_ranks)? (newremote * 2 + 1):(newremote + extra_ranks);

    /* Exchange the data */
    return_value = ompi_coll_base_sendrecv_actual(tmpsend, count, dtype, remote, MCA_COLL_BASE_TAG_ALLREDUCE, tmprecv, count, dtype, remote, MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != return_value) { line = __LINE__; goto error_hndl; }

    /* Apply operation */
    if (rank < remote) {
      /* tmprecv = tmpsend (op) tmprecv */
      ompi_op_reduce(op, tmpsend, tmprecv, count, dtype);
      tmpswap = tmprecv;
      tmprecv = tmpsend;
      tmpsend = tmpswap;
    } else {
      /* tmpsend = tmprecv (op) tmpsend */
      ompi_op_reduce(op, tmprecv, tmpsend, count, dtype);
    }
  }

  /* Handle non-power-of-two case:
      - Odd ranks less than 2 * extra_ranks send result from tmpsend to
      (rank - 1)
      - Even ranks less than 2 * extra_ranks receive result from (rank + 1)
  */
  if (rank < (2 * extra_ranks)) {
    if (0 == (rank % 2)) {
      return_value = MCA_PML_CALL(recv(recieve_buffer, count, dtype, (rank + 1), MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE));
      tmpsend = (char*)recieve_buffer;
    } else {
      return_value = MCA_PML_CALL(send(tmpsend, count, dtype, (rank - 1), MCA_COLL_BASE_TAG_ALLREDUCE, MCA_PML_BASE_SEND_STANDARD, comm));
    }
  }

  /* Ensure that the final result is in recieve_buffer */
  if (tmpsend != recieve_buffer) {
    return_value = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)recieve_buffer, tmpsend);
    if (return_value < 0) { line = __LINE__; goto error_hndl; }
  }

  if (NULL != inplacebuf_free) free(inplacebuf_free);
  return MPI_SUCCESS;
}
