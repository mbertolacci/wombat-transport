MODULE Vdiff_Trace_Mod
  USE Precision_Mod, ONLY : fp
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: Vdiff_Trace_Save, Vdiff_Trace_Get

  REAL(fp), ALLOCATABLE, SAVE :: saved_kvh(:,:,:)
  REAL(fp), ALLOCATABLE, SAVE :: saved_kvm(:,:,:)
  REAL(fp), ALLOCATABLE, SAVE :: saved_tpert(:,:)
  REAL(fp), ALLOCATABLE, SAVE :: saved_qpert(:,:)

CONTAINS

  SUBROUTINE Vdiff_Trace_Save(kvh, kvm, tpert, qpert)
    REAL(fp), INTENT(IN) :: kvh(:,:,:)
    REAL(fp), INTENT(IN) :: kvm(:,:,:)
    REAL(fp), INTENT(IN) :: tpert(:,:)
    REAL(fp), INTENT(IN) :: qpert(:,:)
    IF (ALLOCATED(saved_kvh)) DEALLOCATE(saved_kvh)
    IF (ALLOCATED(saved_kvm)) DEALLOCATE(saved_kvm)
    IF (ALLOCATED(saved_tpert)) DEALLOCATE(saved_tpert)
    IF (ALLOCATED(saved_qpert)) DEALLOCATE(saved_qpert)
    ALLOCATE(saved_kvh(SIZE(kvh,1), SIZE(kvh,2), SIZE(kvh,3)))
    ALLOCATE(saved_kvm(SIZE(kvm,1), SIZE(kvm,2), SIZE(kvm,3)))
    ALLOCATE(saved_tpert(SIZE(tpert,1), SIZE(tpert,2)))
    ALLOCATE(saved_qpert(SIZE(qpert,1), SIZE(qpert,2)))
    saved_kvh = kvh
    saved_kvm = kvm
    saved_tpert = tpert
    saved_qpert = qpert
  END SUBROUTINE Vdiff_Trace_Save

  SUBROUTINE Vdiff_Trace_Get(kvh, kvm, tpert, qpert)
    REAL(fp), INTENT(OUT) :: kvh(:,:,:)
    REAL(fp), INTENT(OUT) :: kvm(:,:,:)
    REAL(fp), INTENT(OUT) :: tpert(:,:)
    REAL(fp), INTENT(OUT) :: qpert(:,:)
    kvh = saved_kvh
    kvm = saved_kvm
    tpert = saved_tpert
    qpert = saved_qpert
  END SUBROUTINE Vdiff_Trace_Get

END MODULE Vdiff_Trace_Mod
