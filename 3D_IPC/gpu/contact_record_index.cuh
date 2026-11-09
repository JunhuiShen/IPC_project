#pragma once

// Header-only record/CSR addressing. Standalone reduction fixtures can keep
// their legacy Work records; production Work supplies the optional accessors.
#if defined(__CUDACC__)
#define IPC_CONTACT_RECORD_HD __host__ __device__
#else
#define IPC_CONTACT_RECORD_HD
#endif
template<class WorkType>
IPC_CONTACT_RECORD_HD inline auto contact_record_slot_impl(const WorkType& work,int pair,int)
    ->decltype(work.record_slot(pair)) {return work.record_slot(pair);}
template<class WorkType>
IPC_CONTACT_RECORD_HD inline int contact_record_slot_impl(const WorkType&,int pair,long) {return pair;}
template<class WorkType>
IPC_CONTACT_RECORD_HD inline int contact_record_slot(const WorkType& work,int pair) {
    return contact_record_slot_impl(work,pair,0);
}
template<class WorkType>
IPC_CONTACT_RECORD_HD inline auto contact_incident_pair_impl(const WorkType& work,std::size_t incident,int)
    ->decltype(work.incident_pair(incident)) {return work.incident_pair(incident);}
template<class WorkType>
IPC_CONTACT_RECORD_HD inline int contact_incident_pair_impl(const WorkType& work,std::size_t incident,long) {
    return work.incidents[incident].contact;
}
template<class WorkType>
IPC_CONTACT_RECORD_HD inline int contact_incident_pair(const WorkType& work,std::size_t incident) {
    return contact_incident_pair_impl(work,incident,0);
}
template<class WorkType>
IPC_CONTACT_RECORD_HD inline auto contact_pending_impl(const WorkType& work,int)
    ->decltype(work.pending_contacts) {return work.pending_contacts;}
template<class WorkType>
IPC_CONTACT_RECORD_HD inline bool contact_pending_impl(const WorkType&,long) {return false;}
template<class WorkType>
IPC_CONTACT_RECORD_HD inline bool contact_pending(const WorkType& work) {return contact_pending_impl(work,0);}
#undef IPC_CONTACT_RECORD_HD
