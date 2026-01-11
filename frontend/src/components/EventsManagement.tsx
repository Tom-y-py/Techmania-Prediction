/**
 * Events Management Component
 * Komponenta pro správu událostí (events) v Plzni
 */

'use client';

import { useState, useEffect } from 'react';
import { useEvents } from '@/hooks/useApi';
import type { EventResponse, EventCreate } from '@/types/api';

interface EventsManagementProps {
  startDate?: string;
  endDate?: string;
}

export default function EventsManagement({ startDate, endDate }: EventsManagementProps) {
  const {
    data,
    loading,
    error,
    fetchEvents,
    createEvent,
    updateEvent,
    deleteEvent,
  } = useEvents();

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [editingEvent, setEditingEvent] = useState<EventResponse | null>(null);
  const [filterCategory, setFilterCategory] = useState<string>('');
  const [filterMinImpact, setFilterMinImpact] = useState<number>(1);

  useEffect(() => {
    loadEvents();
  }, [startDate, endDate, filterCategory, filterMinImpact]);

  const loadEvents = async () => {
    try {
      await fetchEvents({
        start_date: startDate,
        end_date: endDate,
        category: filterCategory || undefined,
        min_impact: filterMinImpact,
        limit: 100,
      });
    } catch (err) {
      console.error('Failed to load events:', err);
    }
  };

  const handleCreateEvent = async (eventData: EventCreate) => {
    try {
      await createEvent(eventData);
      setShowCreateModal(false);
      loadEvents();
    } catch (err) {
      console.error('Failed to create event:', err);
    }
  };

  const handleDeleteEvent = async (eventId: number) => {
    if (!confirm('Opravdu chcete smazat tento event?')) return;

    try {
      await deleteEvent(eventId);
      loadEvents();
    } catch (err) {
      console.error('Failed to delete event:', err);
    }
  };

  const categories = ['obecne', 'hudba', 'sport', 'kultura', 'festival', 'veletrh', 'custom'];
  const attendanceLevels = ['male', 'stredni', 'velke', 'masivni'];

  if (loading && !data) {
    return (
      <div className="flex justify-center items-center p-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Správa událostí</h2>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
        >
          + Přidat event
        </button>
      </div>

      {/* Filters */}
      <div className="flex gap-4 items-end">
        <div className="flex-1">
          <label className="block text-sm font-medium mb-1">Kategorie</label>
          <select
            value={filterCategory}
            onChange={(e) => setFilterCategory(e.target.value)}
            className="w-full px-3 py-2 border rounded-lg"
          >
            <option value="">Všechny kategorie</option>
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                {cat}
              </option>
            ))}
          </select>
        </div>

        <div className="flex-1">
          <label className="block text-sm font-medium mb-1">Minimální dopad</label>
          <select
            value={filterMinImpact}
            onChange={(e) => setFilterMinImpact(Number(e.target.value))}
            className="w-full px-3 py-2 border rounded-lg"
          >
            {[1, 2, 3, 4, 5].map((level) => (
              <option key={level} value={level}>
                {level}+ ⭐
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Events List */}
      {data && (
        <div className="space-y-4">
          <div className="text-sm text-gray-600">
            Celkem: {data.total_count} událostí
          </div>

          <div className="grid gap-4">
            {data.events.map((event) => (
              <div
                key={event.id}
                className="border rounded-lg p-4 hover:shadow-md transition"
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h3 className="font-semibold text-lg">{event.title}</h3>
                      <span className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded">
                        {event.category}
                      </span>
                      <span className="text-xs px-2 py-1 bg-yellow-100 text-yellow-700 rounded">
                        {'⭐'.repeat(event.impact_level)}
                      </span>
                    </div>

                    <p className="text-gray-600 text-sm mb-2">{event.description}</p>

                    <div className="flex gap-4 text-sm text-gray-500">
                      <span>📅 {new Date(event.event_date).toLocaleDateString('cs-CZ')}</span>
                      <span>📍 {event.venue}</span>
                      <span>👥 {event.expected_attendance}</span>
                      <span>🔗 {event.source}</span>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() => setEditingEvent(event)}
                      className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded transition"
                    >
                      Upravit
                    </button>
                    <button
                      onClick={() => handleDeleteEvent(event.id)}
                      className="px-3 py-1 text-sm bg-red-100 text-red-700 hover:bg-red-200 rounded transition"
                    >
                      Smazat
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Create/Edit Modal */}
      {(showCreateModal || editingEvent) && (
        <EventFormModal
          event={editingEvent}
          onClose={() => {
            setShowCreateModal(false);
            setEditingEvent(null);
          }}
          onSave={async (eventData) => {
            if (editingEvent) {
              await updateEvent(editingEvent.id, eventData);
              setEditingEvent(null);
            } else {
              await handleCreateEvent(eventData);
            }
            loadEvents();
          }}
          categories={categories}
          attendanceLevels={attendanceLevels}
        />
      )}
    </div>
  );
}

// Event Form Modal
interface EventFormModalProps {
  event: EventResponse | null;
  onClose: () => void;
  onSave: (event: EventCreate) => Promise<void>;
  categories: string[];
  attendanceLevels: string[];
}

function EventFormModal({
  event,
  onClose,
  onSave,
  categories,
  attendanceLevels,
}: EventFormModalProps) {
  const [formData, setFormData] = useState<EventCreate>({
    event_date: event?.event_date || new Date().toISOString().split('T')[0],
    title: event?.title || '',
    description: event?.description || '',
    venue: event?.venue || 'Plzen',
    category: event?.category || 'obecne',
    expected_attendance: event?.expected_attendance || 'stredni',
    impact_level: event?.impact_level || 2,
  });

  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      await onSave(formData);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <h3 className="text-xl font-bold mb-4">
          {event ? 'Upravit event' : 'Nový event'}
        </h3>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Datum</label>
            <input
              type="date"
              value={formData.event_date}
              onChange={(e) =>
                setFormData({ ...formData, event_date: e.target.value })
              }
              className="w-full px-3 py-2 border rounded-lg"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Název</label>
            <input
              type="text"
              value={formData.title}
              onChange={(e) => setFormData({ ...formData, title: e.target.value })}
              className="w-full px-3 py-2 border rounded-lg"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Popis</label>
            <textarea
              value={formData.description || ''}
              onChange={(e) =>
                setFormData({ ...formData, description: e.target.value })
              }
              className="w-full px-3 py-2 border rounded-lg"
              rows={3}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Kategorie</label>
              <select
                value={formData.category}
                onChange={(e) =>
                  setFormData({ ...formData, category: e.target.value })
                }
                className="w-full px-3 py-2 border rounded-lg"
              >
                {categories.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Místo</label>
              <input
                type="text"
                value={formData.venue}
                onChange={(e) => setFormData({ ...formData, venue: e.target.value })}
                className="w-full px-3 py-2 border rounded-lg"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Návštěvnost</label>
              <select
                value={formData.expected_attendance}
                onChange={(e) =>
                  setFormData({ ...formData, expected_attendance: e.target.value })
                }
                className="w-full px-3 py-2 border rounded-lg"
              >
                {attendanceLevels.map((level) => (
                  <option key={level} value={level}>
                    {level}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Dopad (1-5)</label>
              <input
                type="number"
                min="1"
                max="5"
                value={formData.impact_level}
                onChange={(e) =>
                  setFormData({ ...formData, impact_level: Number(e.target.value) })
                }
                className="w-full px-3 py-2 border rounded-lg"
              />
            </div>
          </div>

          <div className="flex gap-3 pt-4">
            <button
              type="submit"
              disabled={saving}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:opacity-50"
            >
              {saving ? 'Ukládám...' : 'Uložit'}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-gray-200 rounded-lg hover:bg-gray-300 transition"
            >
              Zrušit
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
